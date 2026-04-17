# iddpg_agent.py - Independent DDPG多智能体
"""
IDDPG: 每个agent独立训练，互不共享信息
每个agent有独立的replay buffer和网络
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from iddpg_model import IDDPGActor, IDDPGCritic
from config import Config


class ReplayBuffer:
    """单agent经验回放缓冲区"""
    
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    
    def push(self, obs, action, reward, next_obs, done):
        experience = (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done)
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []
        
        for idx in indices:
            obs, action, reward, next_obs, done = self.buffer[idx]
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)
        
        return (
            torch.FloatTensor(np.array(obs_list)).to(device),
            torch.FloatTensor(np.array(action_list)).to(device),
            torch.FloatTensor(np.array(reward_list)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_obs_list)).to(device),
            torch.FloatTensor(np.array(done_list)).unsqueeze(1).to(device)
        )
    
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


class IDDPGAgent:
    """Independent DDPG多智能体"""
    
    def __init__(self, num_agents, obs_dim, action_dim, device):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 超参数
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.batch_size = Config.BATCH_SIZE
        
        # 每个agent独立的网络
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        self.replay_buffers = []  # 每个agent独立的buffer
        self.noises = []
        
        for i in range(num_agents):
            # Actor
            actor = IDDPGActor(obs_dim, action_dim).to(device)
            actor_target = IDDPGActor(obs_dim, action_dim).to(device)
            actor_target.load_state_dict(actor.state_dict())
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=Config.ACTOR_LR))
            
            # Critic（只看自己的观测和动作）
            critic = IDDPGCritic(obs_dim, action_dim).to(device)
            critic_target = IDDPGCritic(obs_dim, action_dim).to(device)
            critic_target.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=Config.CRITIC_LR))
            
            # 独立的replay buffer
            replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE, obs_dim, action_dim)
            self.replay_buffers.append(replay_buffer)
            
            # 噪声
            noise = OUNoise(action_dim, sigma=Config.NOISE_SCALE)
            self.noises.append(noise)
        
        self.noise_scale = Config.NOISE_SCALE
        self.total_steps = 0
    
    def select_actions(self, obs_list, eval_mode=False):
        """选择所有agent的动作"""
        actions = []
        
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            
            if not eval_mode:
                noise = self.noises[i].sample() * self.noise_scale
                action = np.clip(action + noise, -1.0, 1.0)
            
            actions.append(action)
        
        return actions
    
    def push_experience(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        """将经验存入各自的buffer"""
        for i in range(self.num_agents):
            self.replay_buffers[i].push(
                obs_list[i], action_list[i], reward_list[i], 
                next_obs_list[i], done_list[i]
            )
    
    def update(self):
        """独立更新每个agent"""
        actor_losses = []
        critic_losses = []
        
        for agent_idx in range(self.num_agents):
            # 检查buffer是否足够
            if len(self.replay_buffers[agent_idx]) < self.batch_size:
                actor_losses.append(0.0)
                critic_losses.append(0.0)
                continue
            
            # 采样（只从自己的buffer）
            obs, actions, rewards, next_obs, dones = \
                self.replay_buffers[agent_idx].sample(self.batch_size, self.device)
            
            # ========== 更新Critic ==========
            with torch.no_grad():
                next_actions = self.actor_targets[agent_idx](next_obs)
                target_q = self.critic_targets[agent_idx](next_obs, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q = self.critics[agent_idx](obs, actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
            self.critic_optimizers[agent_idx].step()
            
            # ========== 更新Actor ==========
            actor_actions = self.actors[agent_idx](obs)
            actor_loss = -self.critics[agent_idx](obs, actor_actions).mean()
            
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
            self.actor_optimizers[agent_idx].step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
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
        """重置噪声"""
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


if __name__ == '__main__':
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    Config.print_obs_dims()
    
    num_agents = Config.NUM_AGENTS
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    
    agent = IDDPGAgent(num_agents, obs_dim, action_dim, device)
    print(f"\n✅ IDDPG Agent创建成功!")
    print(f"   {num_agents}个独立的Actor-Critic")
    print(f"   {num_agents}个独立的Replay Buffer")

