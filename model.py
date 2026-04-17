# model.py - MADDPG网络架构
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


class Actor(nn.Module):
    """
    MADDPG Actor网络
    输入：单个agent的局部观测
    输出：连续动作 [-1, 1]
    
    特点：分布式执行，只使用自己的观测
    """
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 计算观测各部分维度
        self.global_dim = Config.GOAL_REL_DIM + Config.GOAL_DIST_DIM + Config.SELF_VEL_DIM  # 7
        self.ray_dim = Config.NUM_RAY_DIRECTIONS * Config.RAY_INFO_PER_DIR  # 52
        self.local_goal_dim = Config.LOCAL_GOAL_DIM  # 5
        self.dyn_dim = Config.NUM_DYNAMIC_OBSTACLES * Config.DYN_OBS_INFO_DIM  # 35
        self.hist_dim = Config.ACTION_HISTORY_LEN  # 5
        self.other_agents_dim = (Config.NUM_AGENTS - 1) * Config.OTHER_AGENT_INFO_DIM  # (n-1)*7
        
        # Ray特征编码器（Transformer）
        self.ray_embed = nn.Linear(Config.RAY_INFO_PER_DIR, 64)
        self.pos_encoder = PositionalEncoding(64, max_len=Config.NUM_RAY_DIRECTIONS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 全局特征编码器
        self.global_fc = nn.Linear(self.global_dim, 128)
        self.global_ln = nn.LayerNorm(128)
        
        # 局部目标编码器
        self.local_goal_fc = nn.Linear(self.local_goal_dim, 64)
        self.local_goal_ln = nn.LayerNorm(64)
        
        # 动态障碍编码器
        self.dyn_fc = nn.Linear(self.dyn_dim, 128)
        self.dyn_ln = nn.LayerNorm(128)
        
        # 历史编码器
        self.hist_fc = nn.Linear(self.hist_dim, 64)
        self.hist_ln = nn.LayerNorm(64)
        
        # 其他agent信息编码器
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
        
        # 解析观测各部分
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
        
        # Ray特征处理（Transformer）
        ray_feat = ray_feat.view(batch_size, Config.NUM_RAY_DIRECTIONS, Config.RAY_INFO_PER_DIR)
        ray_embed = self.ray_embed(ray_feat)
        ray_embed = self.pos_encoder(ray_embed)
        ray_encoded = self.transformer(ray_embed)
        ray_encoded = ray_encoded.view(batch_size, -1)
        
        # 其他特征处理
        global_encoded = F.relu(self.global_ln(self.global_fc(global_feat)))
        local_goal_encoded = F.relu(self.local_goal_ln(self.local_goal_fc(local_goal_feat)))
        dyn_encoded = F.relu(self.dyn_ln(self.dyn_fc(dyn_feat)))
        hist_encoded = F.relu(self.hist_ln(self.hist_fc(hist_feat)))
        
        # 其他agent信息处理
        if self.other_agents_dim > 0:
            other_agents_feat = obs[:, idx:idx + self.other_agents_dim]
            other_agents_encoded = F.relu(self.other_agents_ln(self.other_agents_fc(other_agents_feat)))
            fused = torch.cat([ray_encoded, global_encoded, local_goal_encoded, 
                              dyn_encoded, hist_encoded, other_agents_encoded], dim=1)
        else:
            fused = torch.cat([ray_encoded, global_encoded, local_goal_encoded, 
                              dyn_encoded, hist_encoded], dim=1)
        
        # 融合层
        x = F.relu(self.fusion_ln1(self.fusion1(fused)))
        identity = x
        x = F.relu(self.fusion_ln2(self.fusion2(x)))
        x = x + identity  # 残差连接
        x = F.relu(self.fusion_ln3(self.fusion3(x)))
        
        # 输出动作
        action = torch.tanh(self.output(x))
        return action


class Critic(nn.Module):
    """
    MADDPG Critic网络
    输入：所有agent的观测 + 所有agent的动作
    输出：Q值
    
    特点：集中式训练，可以访问全局信息
    """
    def __init__(self, num_agents, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 总输入维度：所有观测 + 所有动作
        total_obs_dim = obs_dim * num_agents
        total_action_dim = action_dim * num_agents
        total_input_dim = total_obs_dim + total_action_dim
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(total_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, hidden_size),
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
    
    def forward(self, obs_list, action_list):
        """
        Args:
            obs_list: [batch, num_agents * obs_dim] 或 List[Tensor]
            action_list: [batch, num_agents * action_dim] 或 List[Tensor]
        """
        # 如果输入是列表，拼接成tensor
        if isinstance(obs_list, list):
            obs = torch.cat(obs_list, dim=1)
        else:
            obs = obs_list
        
        if isinstance(action_list, list):
            actions = torch.cat(action_list, dim=1)
        else:
            actions = action_list
        
        # 编码
        obs_encoded = self.obs_encoder(obs)
        action_encoded = self.action_encoder(actions)
        
        # 融合
        fused = torch.cat([obs_encoded, action_encoded], dim=1)
        q_value = self.fusion(fused)
        
        return q_value


class MADDPGNetwork:
    """
    MADDPG网络管理器
    管理所有agent的Actor和Critic网络
    """
    def __init__(self, num_agents, obs_dim, action_dim, device):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 每个agent有自己的Actor和Critic
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        
        for i in range(num_agents):
            # Actor
            actor = Actor(obs_dim, action_dim).to(device)
            actor_target = Actor(obs_dim, action_dim).to(device)
            actor_target.load_state_dict(actor.state_dict())
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            
            # Critic（每个agent一个，但输入是全局信息）
            critic = Critic(num_agents, obs_dim, action_dim).to(device)
            critic_target = Critic(num_agents, obs_dim, action_dim).to(device)
            critic_target.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
    
    def get_actions(self, obs_list, eval_mode=False):
        """获取所有agent的动作"""
        actions = []
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            actions.append(action)
        return actions
    
    def soft_update(self, tau):
        """软更新所有目标网络"""
        for i in range(self.num_agents):
            for src_param, tgt_param in zip(self.actors[i].parameters(), 
                                            self.actor_targets[i].parameters()):
                tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
            
            for src_param, tgt_param in zip(self.critics[i].parameters(), 
                                            self.critic_targets[i].parameters()):
                tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
    
    def save(self, path):
        """保存所有网络"""
        state_dict = {
            'num_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }
        for i in range(self.num_agents):
            state_dict[f'actor_{i}'] = self.actors[i].state_dict()
            state_dict[f'actor_target_{i}'] = self.actor_targets[i].state_dict()
            state_dict[f'critic_{i}'] = self.critics[i].state_dict()
            state_dict[f'critic_target_{i}'] = self.critic_targets[i].state_dict()
        
        torch.save(state_dict, path)
    
    def load(self, path):
        """加载所有网络"""
        state_dict = torch.load(path, map_location=self.device)
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state_dict[f'actor_{i}'])
            self.actor_targets[i].load_state_dict(state_dict[f'actor_target_{i}'])
            self.critics[i].load_state_dict(state_dict[f'critic_{i}'])
            self.critic_targets[i].load_state_dict(state_dict[f'critic_target_{i}'])


if __name__ == '__main__':
    # 测试网络
    Config.print_obs_dims()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"Agent数量: {num_agents}")
    
    # 测试Actor
    actor = Actor(obs_dim, action_dim).to(device)
    test_obs = torch.randn(4, obs_dim).to(device)
    action = actor(test_obs)
    print(f"\nActor输出形状: {action.shape}")
    
    # 测试Critic
    critic = Critic(num_agents, obs_dim, action_dim).to(device)
    test_obs_all = torch.randn(4, obs_dim * num_agents).to(device)
    test_actions_all = torch.randn(4, action_dim * num_agents).to(device)
    q_value = critic(test_obs_all, test_actions_all)
    print(f"Critic输出形状: {q_value.shape}")
    
    # 测试网络管理器
    network = MADDPGNetwork(num_agents, obs_dim, action_dim, device)
    test_obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
    actions = network.get_actions(test_obs_list)
    print(f"\n网络管理器输出动作数量: {len(actions)}")
    print(f"每个动作形状: {actions[0].shape}")
