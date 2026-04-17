# mappo_agent.py - MAPPO多智能体
"""
MAPPO: Multi-Agent PPO
- On-policy: 收集轨迹→更新→清空
- 使用GAE计算优势
- PPO裁剪更新
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mappo_model import MAPPOActor, MAPPOCritic
from config import Config


class RolloutBuffer:
    """轨迹缓冲区（On-policy）"""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.obs_all = []  # 全局状态（用于Critic）
    
    def push(self, obs, obs_all, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.obs_all.append(obs_all)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.obs.clear()
        self.obs_all.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.obs)


class MAPPOAgent:
    """MAPPO多智能体"""
    
    def __init__(self, num_agents, obs_dim, action_dim, device):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 超参数
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        self.ppo_epoch = Config.PPO_EPOCH
        self.ppo_clip = Config.PPO_CLIP
        self.value_coef = Config.VALUE_COEF
        self.entropy_coef = Config.ENTROPY_COEF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        
        # 每个agent独立的Actor
        self.actors = []
        self.actor_optimizers = []
        
        for i in range(num_agents):
            actor = MAPPOActor(obs_dim, action_dim).to(device)
            optimizer = optim.Adam(actor.parameters(), lr=Config.PPO_LR)
            self.actors.append(actor)
            self.actor_optimizers.append(optimizer)
        
        # 共享的集中式Critic
        self.critic = MAPPOCritic(num_agents, obs_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Config.PPO_LR)
        
        # 轨迹缓冲区（每个agent一个）
        self.buffers = [RolloutBuffer() for _ in range(num_agents)]
        
        self.total_steps = 0
    
    def select_actions(self, obs_list, eval_mode=False):
        """选择所有agent的动作"""
        actions = []
        log_probs = []
        values = []
        
        # 全局状态（所有agent的观测拼接）
        obs_all = np.concatenate(obs_list)
        obs_all_tensor = torch.FloatTensor(obs_all).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Critic评估价值
            value = self.critic(obs_all_tensor)
        
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if eval_mode:
                # 评估模式：使用均值
                with torch.no_grad():
                    mean, _ = self.actors[i](obs_tensor)
                    action = mean.cpu().numpy()[0]
                log_prob = None
            else:
                # 训练模式：采样
                with torch.no_grad():
                    action_tensor, log_prob_tensor = self.actors[i].get_action_log_prob(obs_tensor)
                    action = action_tensor.cpu().numpy()[0]
                    log_prob = log_prob_tensor.cpu().numpy()[0, 0]
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.cpu().numpy()[0, 0])
        
        return actions, log_probs, values
    
    def push_experience(self, obs_list, actions, rewards, log_probs, values, dones):
        """将经验存入各自的buffer"""
        obs_all = np.concatenate(obs_list)
        
        for i in range(self.num_agents):
            self.buffers[i].push(
                obs_list[i], obs_all, actions[i], rewards[i], 
                values[i], log_probs[i], dones[i]
            )
    
    def compute_gae(self, buffer, next_value):
        """计算GAE优势"""
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_obs_list):
        """PPO更新"""
        # 计算下一步的价值（用于GAE）
        next_obs_all = np.concatenate(next_obs_list)
        next_obs_all_tensor = torch.FloatTensor(next_obs_all).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_values = self.critic(next_obs_all_tensor).cpu().numpy()
        
        # 为每个agent计算GAE
        all_advantages = []
        all_returns = []
        
        for i in range(self.num_agents):
            if len(self.buffers[i]) == 0:
                continue
            
            advantages, returns = self.compute_gae(self.buffers[i], next_values[0, 0])
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # 准备训练数据
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for agent_idx in range(self.num_agents):
            if len(self.buffers[agent_idx]) == 0:
                continue
            
            buffer = self.buffers[agent_idx]
            advantages = all_advantages[agent_idx]
            returns = all_returns[agent_idx]
            
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 转为Tensor
            obs_tensor = torch.FloatTensor(np.array(buffer.obs)).to(self.device)
            obs_all_tensor = torch.FloatTensor(np.array(buffer.obs_all)).to(self.device)
            actions_tensor = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(np.array(buffer.log_probs)).unsqueeze(1).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
            returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
            
            # PPO多轮更新
            for _ in range(self.ppo_epoch):
                # 评估当前动作
                new_log_probs, entropy = self.actors[agent_idx].evaluate_actions(obs_tensor, actions_tensor)
                
                # 计算ratio
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                
                # PPO裁剪损失
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # 更新Actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.max_grad_norm)
                self.actor_optimizers[agent_idx].step()
                
                # 更新Critic
                values = self.critic(obs_all_tensor)
                critic_loss = F.mse_loss(values, returns_tensor)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # 清空buffer
        for buffer in self.buffers:
            buffer.clear()
        
        self.total_steps += 1
        
        avg_actor_loss = total_actor_loss / max(num_updates, 1)
        avg_critic_loss = total_critic_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def save_model(self, path):
        """保存模型"""
        state_dict = {
            'num_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'total_steps': self.total_steps,
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
        for i in range(self.num_agents):
            state_dict[f'actor_{i}'] = self.actors[i].state_dict()
            state_dict[f'actor_optimizer_{i}'] = self.actor_optimizers[i].state_dict()
        
        torch.save(state_dict, path)
    
    def load_model(self, path):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)
        
        self.total_steps = state_dict.get('total_steps', 0)
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state_dict[f'actor_{i}'])
            self.actor_optimizers[i].load_state_dict(state_dict[f'actor_optimizer_{i}'])


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    Config.print_obs_dims()
    
    num_agents = Config.NUM_AGENTS
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    
    agent = MAPPOAgent(num_agents, obs_dim, action_dim, device)
    print(f"\n✅ MAPPO Agent创建成功!")
    print(f"   {num_agents}个独立Actor + 1个集中式Critic")

