# iddpg_model.py - Independent DDPG网络架构
"""
IDDPG: 每个agent完全独立训练，把其他agent视为环境的一部分
与MADDPG的区别：Critic只看自己的观测+动作，不看全局
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import Config


class PositionalEncoding(nn.Module):
    """位置编码（用于Transformer）"""
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


class IDDPGActor(nn.Module):
    """
    IDDPG Actor: 与MADDPG Actor相同，使用局部观测
    输入：单个agent的观测（包含其他agent信息）
    输出：连续动作 [-1, 1]
    """
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 计算观测各部分维度
        self.global_dim = Config.GOAL_REL_DIM + Config.GOAL_DIST_DIM + Config.SELF_VEL_DIM
        self.ray_dim = Config.NUM_RAY_DIRECTIONS * Config.RAY_INFO_PER_DIR
        self.local_goal_dim = Config.LOCAL_GOAL_DIM
        self.dyn_dim = Config.NUM_DYNAMIC_OBSTACLES * Config.DYN_OBS_INFO_DIM
        self.hist_dim = Config.ACTION_HISTORY_LEN
        self.other_agents_dim = (Config.NUM_AGENTS - 1) * Config.OTHER_AGENT_INFO_DIM
        
        # Ray特征编码器（Transformer）
        self.ray_embed = nn.Linear(Config.RAY_INFO_PER_DIR, 64)
        self.pos_encoder = PositionalEncoding(64, max_len=Config.NUM_RAY_DIRECTIONS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 其他特征编码器
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
        self.output = nn.Linear(128, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs):
        batch_size = obs.size(0)
        
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
        
        # Ray处理
        ray_feat = ray_feat.view(batch_size, Config.NUM_RAY_DIRECTIONS, Config.RAY_INFO_PER_DIR)
        ray_embed = self.ray_embed(ray_feat)
        ray_embed = self.pos_encoder(ray_embed)
        ray_encoded = self.transformer(ray_embed)
        ray_encoded = ray_encoded.view(batch_size, -1)
        
        # 其他特征
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
        
        x = F.relu(self.fusion_ln1(self.fusion1(fused)))
        identity = x
        x = F.relu(self.fusion_ln2(self.fusion2(x)))
        x = x + identity
        x = F.relu(self.fusion_ln3(self.fusion3(x)))
        
        action = torch.tanh(self.output(x))
        return action


class IDDPGCritic(nn.Module):
    """
    IDDPG Critic: 只看单个agent的观测+动作（独立训练）
    输入：单个agent的obs + 单个agent的action
    输出：Q值
    """
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, action):
        """
        Args:
            obs: [batch, obs_dim] 单个agent的观测
            action: [batch, action_dim] 单个agent的动作
        """
        obs_encoded = self.obs_encoder(obs)
        action_encoded = self.action_encoder(action)
        fused = torch.cat([obs_encoded, action_encoded], dim=1)
        q_value = self.fusion(fused)
        return q_value


if __name__ == '__main__':
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    Config.print_obs_dims()
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    
    print(f"\n测试IDDPG网络:")
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    
    # 测试Actor
    actor = IDDPGActor(obs_dim, action_dim).to(device)
    test_obs = torch.randn(4, obs_dim).to(device)
    action = actor(test_obs)
    print(f"Actor输出: {action.shape}")
    
    # 测试Critic
    critic = IDDPGCritic(obs_dim, action_dim).to(device)
    q_value = critic(test_obs, action)
    print(f"Critic输出: {q_value.shape}")

