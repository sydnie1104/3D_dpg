# agent.py - MADDPG多智能体训练
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from model import Actor, Critic, MADDPGNetwork
from config import Config


class MultiAgentReplayBuffer:
    """多智能体经验回放缓冲区"""
    
    def __init__(self, capacity, num_agents, obs_dim, action_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.buffer = []
        self.pos = 0
    
    def push(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        """
        添加一条经验
        
        Args:
            obs_list: List[np.array], 每个agent的观测
            action_list: List[np.array], 每个agent的动作
            reward_list: List[float], 每个agent的奖励
            next_obs_list: List[np.array], 每个agent的下一观测
            done_list: List[bool], 每个agent是否结束
        """
        experience = (
            [np.array(o, dtype=np.float32) for o in obs_list],
            [np.array(a, dtype=np.float32) for a in action_list],
            [float(r) for r in reward_list],
            [np.array(o, dtype=np.float32) for o in next_obs_list],
            [float(d) for d in done_list]
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, device):
        """采样一批经验"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # 初始化批次数据
        obs_batch = [[] for _ in range(self.num_agents)]
        action_batch = [[] for _ in range(self.num_agents)]
        reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]
        done_batch = [[] for _ in range(self.num_agents)]
        
        for idx in indices:
            obs_list, action_list, reward_list, next_obs_list, done_list = self.buffer[idx]
            
            for i in range(self.num_agents):
                obs_batch[i].append(obs_list[i])
                action_batch[i].append(action_list[i])
                reward_batch[i].append(reward_list[i])
                next_obs_batch[i].append(next_obs_list[i])
                done_batch[i].append(done_list[i])
        
        # 转换为Tensor
        obs_tensors = [torch.FloatTensor(np.array(obs_batch[i])).to(device) 
                       for i in range(self.num_agents)]
        action_tensors = [torch.FloatTensor(np.array(action_batch[i])).to(device) 
                          for i in range(self.num_agents)]
        reward_tensors = [torch.FloatTensor(np.array(reward_batch[i])).unsqueeze(1).to(device) 
                          for i in range(self.num_agents)]
        next_obs_tensors = [torch.FloatTensor(np.array(next_obs_batch[i])).to(device) 
                            for i in range(self.num_agents)]
        done_tensors = [torch.FloatTensor(np.array(done_batch[i])).unsqueeze(1).to(device) 
                        for i in range(self.num_agents)]
        
        return obs_tensors, action_tensors, reward_tensors, next_obs_tensors, done_tensors
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""
    
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class MADDPGAgent:
    """MADDPG多智能体"""
    
    def __init__(self, num_agents, obs_dim, action_dim, device):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 超参数
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.batch_size = Config.BATCH_SIZE
        
        # 网络
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(num_agents):
            # Actor
            actor = Actor(obs_dim, action_dim).to(device)
            actor_target = Actor(obs_dim, action_dim).to(device)
            actor_target.load_state_dict(actor.state_dict())
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=Config.ACTOR_LR))
            
            # Critic
            critic = Critic(num_agents, obs_dim, action_dim).to(device)
            critic_target = Critic(num_agents, obs_dim, action_dim).to(device)
            critic_target.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=Config.CRITIC_LR))
        
        # 经验回放
        self.replay_buffer = MultiAgentReplayBuffer(
            Config.REPLAY_BUFFER_SIZE,
            num_agents,
            obs_dim,
            action_dim
        )
        
        # 噪声
        self.noises = [OUNoise(action_dim, sigma=Config.NOISE_SCALE) for _ in range(num_agents)]
        self.noise_scale = Config.NOISE_SCALE
        
        # 训练统计
        self.total_steps = 0
    
    def select_actions(self, obs_list, eval_mode=False):
        """选择所有agent的动作"""
        actions = []
        
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            
            # 训练模式添加噪声
            if not eval_mode:
                noise = self.noises[i].sample() * self.noise_scale
                action = np.clip(action + noise, -1.0, 1.0)
            
            actions.append(action)
        
        return actions
    
    def update(self):
        """MADDPG更新"""
        if len(self.replay_buffer) < self.batch_size:
            return [0.0] * self.num_agents, [0.0] * self.num_agents
        
        # 采样
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size, self.device)
        
        actor_losses = []
        critic_losses = []
        
        # 更新每个agent
        for agent_idx in range(self.num_agents):
            # ========== 更新Critic ==========
            # 计算目标Q值
            with torch.no_grad():
                # 获取所有agent的目标动作
                next_actions = []
                for i in range(self.num_agents):
                    next_action = self.actor_targets[i](next_obs_batch[i])
                    next_actions.append(next_action)
                
                # 拼接所有观测和动作
                next_obs_all = torch.cat(next_obs_batch, dim=1)
                next_actions_all = torch.cat(next_actions, dim=1)
                
                # 计算目标Q值
                target_q = self.critic_targets[agent_idx](next_obs_all, next_actions_all)
                target_q = reward_batch[agent_idx] + (1 - done_batch[agent_idx]) * self.gamma * target_q
            
            # 计算当前Q值
            obs_all = torch.cat(obs_batch, dim=1)
            actions_all = torch.cat(action_batch, dim=1)
            current_q = self.critics[agent_idx](obs_all, actions_all)
            
            # Critic损失
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
            self.critic_optimizers[agent_idx].step()
            
            critic_losses.append(critic_loss.item())
            
            # ========== 更新Actor ==========
            # 获取当前agent的动作（保持梯度）
            current_agent_action = self.actors[agent_idx](obs_batch[agent_idx])
            
            # 构建动作列表（只有当前agent的动作需要梯度）
            all_actions = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    all_actions.append(current_agent_action)
                else:
                    all_actions.append(action_batch[i].detach())
            
            all_actions_cat = torch.cat(all_actions, dim=1)
            
            # Actor损失（最大化Q值）
            actor_loss = -self.critics[agent_idx](obs_all, all_actions_cat).mean()
            
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
            self.actor_optimizers[agent_idx].step()
            
            actor_losses.append(actor_loss.item())
        
        # 软更新目标网络
        self._soft_update_all()
        
        self.total_steps += 1
        
        return actor_losses, critic_losses
    
    def _soft_update_all(self):
        """软更新所有目标网络"""
        for i in range(self.num_agents):
            for src_param, tgt_param in zip(self.actors[i].parameters(), 
                                            self.actor_targets[i].parameters()):
                tgt_param.data.copy_(self.tau * src_param.data + (1 - self.tau) * tgt_param.data)
            
            for src_param, tgt_param in zip(self.critics[i].parameters(), 
                                            self.critic_targets[i].parameters()):
                tgt_param.data.copy_(self.tau * src_param.data + (1 - self.tau) * tgt_param.data)
    
    def decay_noise(self):
        """衰减噪声"""
        self.noise_scale = max(Config.NOISE_MIN, self.noise_scale * Config.NOISE_DECAY)
        for noise in self.noises:
            noise.sigma = self.noise_scale
    
    def reset_noise(self):
        """重置噪声状态"""
        for noise in self.noises:
            noise.reset()
    
    def save_model(self, path):
        """保存模型"""
        state_dict = {
            'num_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'total_steps': self.total_steps,
            'noise_scale': self.noise_scale,
        }
        
        for i in range(self.num_agents):
            state_dict[f'actor_{i}'] = self.actors[i].state_dict()
            state_dict[f'actor_target_{i}'] = self.actor_targets[i].state_dict()
            state_dict[f'critic_{i}'] = self.critics[i].state_dict()
            state_dict[f'critic_target_{i}'] = self.critic_targets[i].state_dict()
            state_dict[f'actor_optimizer_{i}'] = self.actor_optimizers[i].state_dict()
            state_dict[f'critic_optimizer_{i}'] = self.critic_optimizers[i].state_dict()
        
        torch.save(state_dict, path)
    
    def load_model(self, path):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)
        
        self.total_steps = state_dict.get('total_steps', 0)
        self.noise_scale = state_dict.get('noise_scale', Config.NOISE_SCALE)
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state_dict[f'actor_{i}'])
            self.actor_targets[i].load_state_dict(state_dict[f'actor_target_{i}'])
            self.critics[i].load_state_dict(state_dict[f'critic_{i}'])
            self.critic_targets[i].load_state_dict(state_dict[f'critic_target_{i}'])
            self.actor_optimizers[i].load_state_dict(state_dict[f'actor_optimizer_{i}'])
            self.critic_optimizers[i].load_state_dict(state_dict[f'critic_optimizer_{i}'])


# 兼容旧接口
DDPGAgent = MADDPGAgent


if __name__ == '__main__':
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    Config.print_obs_dims()
    
    num_agents = Config.NUM_AGENTS
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    
    agent = MADDPGAgent(num_agents, obs_dim, action_dim, device)
    
    # 模拟一些经验
    for _ in range(200):
        obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
        action_list = [np.random.randn(action_dim) for _ in range(num_agents)]
        reward_list = [np.random.randn() for _ in range(num_agents)]
        next_obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
        done_list = [False for _ in range(num_agents)]
        
        agent.replay_buffer.push(obs_list, action_list, reward_list, next_obs_list, done_list)
    
    # 测试更新
    actor_losses, critic_losses = agent.update()
    print(f"\nActor losses: {actor_losses}")
    print(f"Critic losses: {critic_losses}")
    
    # 测试选择动作
    obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
    actions = agent.select_actions(obs_list)
    print(f"\n选择的动作: {[a.shape for a in actions]}")
