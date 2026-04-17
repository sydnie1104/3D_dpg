# mappo_model.py - MAPPO网络架构
"""
MAPPO: Multi-Agent PPO
- Actor: 输出高斯分布（mean, std）
- Critic: 集中式价值函数（看全局状态）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import Config


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MAPPOActor(nn.Module):
    """
    MAPPO Actor: 输出高斯分布参数
    输入：单个agent的局部观测
    输出：mean, std（用于采样动作）
    """
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 观测编码（与MADDPG相同）
        self.global_dim = Config.GOAL_REL_DIM + Config.GOAL_DIST_DIM + Config.SELF_VEL_DIM
        self.ray_dim = Config.NUM_RAY_DIRECTIONS * Config.RAY_INFO_PER_DIR
        self.local_goal_dim = Config.LOCAL_GOAL_DIM
        self.dyn_dim = Config.NUM_DYNAMIC_OBSTACLES * Config.DYN_OBS_INFO_DIM
        self.hist_dim = Config.ACTION_HISTORY_LEN
        self.other_agents_dim = (Config.NUM_AGENTS - 1) * Config.OTHER_AGENT_INFO_DIM
        
        # Ray编码器
        self.ray_embed = nn.Linear(Config.RAY_INFO_PER_DIR, 64)
        self.pos_encoder = PositionalEncoding(64, max_len=Config.NUM_RAY_DIRECTIONS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 其他编码器
        self.global_fc = nn.Linear(self.global_dim, 128)
        self.global_ln = nn.LayerNorm(128)
        self.local_goal_fc = nn.Linear(self.local_goal_dim, 64)
        self.local_goal_ln = nn.LayerNorm(64)
        self.dyn_fc = nn.Linear(self.dyn_dim, 128)
        self.dyn_ln = nn.LayerNorm(128)
        self.hist_fc = nn.Linear(self.hist_dim, 64)
        self.hist_ln = nn.LayerNorm(64)
        
        if self.other_agents_dim > 0:
            self.other_agents_fc = nn.Linear(self.other_agents_dim, 128)
            self.other_agents_ln = nn.LayerNorm(128)
            fusion_dim = 64 * Config.NUM_RAY_DIRECTIONS + 128 + 64 + 128 + 64 + 128
        else:
            fusion_dim = 64 * Config.NUM_RAY_DIRECTIONS + 128 + 64 + 128 + 64
        
        # 融合层
        self.fusion1 = nn.Linear(fusion_dim, hidden_size)
        self.fusion_ln1 = nn.LayerNorm(hidden_size)
        self.fusion2 = nn.Linear(hidden_size, hidden_size)
        self.fusion_ln2 = nn.LayerNorm(hidden_size)
        self.fusion3 = nn.Linear(hidden_size, 128)
        self.fusion_ln3 = nn.LayerNorm(128)
        
        # 输出层：均值和标准差
        self.mean_fc = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的log(std)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.mean_fc.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs):
        batch_size = obs.size(0)
        
        # 解析观测
        idx = 0
        global_feat = obs[:, idx:idx + self.global_dim]
        idx += self.global_dim
        ray_feat = obs[:, idx:idx + self.ray_dim]
        idx += self.ray_dim
        local_goal_feat = obs[:, idx:idx + self.local_goal_dim]
        idx += self.local_goal_dim
        dyn_feat = obs[:, idx:idx + self.dyn_dim]
        idx += self.dyn_dim
        hist_feat = obs[:, idx:idx + self.hist_dim]
        idx += self.hist_dim
        
        # 编码
        ray_feat = ray_feat.view(batch_size, Config.NUM_RAY_DIRECTIONS, Config.RAY_INFO_PER_DIR)
        ray_embed = self.ray_embed(ray_feat)
        ray_embed = self.pos_encoder(ray_embed)
        ray_encoded = self.transformer(ray_embed).view(batch_size, -1)
        
        global_encoded = F.relu(self.global_ln(self.global_fc(global_feat)))
        local_goal_encoded = F.relu(self.local_goal_ln(self.local_goal_fc(local_goal_feat)))
        dyn_encoded = F.relu(self.dyn_ln(self.dyn_fc(dyn_feat)))
        hist_encoded = F.relu(self.hist_ln(self.hist_fc(hist_feat)))
        
        if self.other_agents_dim > 0:
            other_agents_feat = obs[:, idx:idx + self.other_agents_dim]
            other_agents_encoded = F.relu(self.other_agents_ln(self.other_agents_fc(other_agents_feat)))
            fused = torch.cat([ray_encoded, global_encoded, local_goal_encoded, 
                              dyn_encoded, hist_encoded, other_agents_encoded], dim=1)
        else:
            fused = torch.cat([ray_encoded, global_encoded, local_goal_encoded, 
                              dyn_encoded, hist_encoded], dim=1)
        
        # 融合
        x = F.relu(self.fusion_ln1(self.fusion1(fused)))
        identity = x
        x = F.relu(self.fusion_ln2(self.fusion2(x)))
        x = x + identity
        x = F.relu(self.fusion_ln3(self.fusion3(x)))
        
        # 输出均值和标准差
        mean = torch.tanh(self.mean_fc(x))  # [-1, 1]
        std = torch.exp(self.log_std).expand_as(mean)  # 确保std > 0
        
        return mean, std
    
    def get_action_log_prob(self, obs):
        """采样动作并返回log概率"""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """评估给定动作的log概率和熵"""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    MAPPO Critic: 集中式价值函数
    输入：所有agent的观测（全局状态）
    输出：价值估计
    """
    def __init__(self, num_agents, obs_dim, hidden_size=256):
        super().__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        total_obs_dim = obs_dim * num_agents
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(total_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs_all):
        """
        Args:
            obs_all: [batch, num_agents * obs_dim] 所有agent的观测
        """
        value = self.encoder(obs_all)
        return value


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    Config.print_obs_dims()
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    print(f"\n测试MAPPO网络:")
    
    # 测试Actor
    actor = MAPPOActor(obs_dim, action_dim).to(device)
    test_obs = torch.randn(4, obs_dim).to(device)
    mean, std = actor(test_obs)
    print(f"Actor输出 - mean: {mean.shape}, std: {std.shape}")
    
    action, log_prob = actor.get_action_log_prob(test_obs)
    print(f"采样动作: {action.shape}, log_prob: {log_prob.shape}")
    
    # 测试Critic
    critic = MAPPOCritic(num_agents, obs_dim).to(device)
    test_obs_all = torch.randn(4, obs_dim * num_agents).to(device)
    value = critic(test_obs_all)
    print(f"Critic输出: {value.shape}")

