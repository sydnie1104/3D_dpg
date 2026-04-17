# test_multi_algo.py - 通用多算法测试脚本
"""
支持测试 MADDPG, MAPPO, IDDPG 三种算法
使用方法：
    python test_multi_algo.py --algo MADDPG
    python test_multi_algo.py --algo MAPPO
    python test_multi_algo.py --algo IDDPG
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

import os
import argparse
import numpy as np
from env import MultiAgentDroneEnv
from config import Config


def load_agent(algo, num_agents, obs_dim, action_dim, device):
    """根据算法类型加载对应的agent"""
    if algo == "MADDPG":
        from agent import MADDPGAgent
        agent = MADDPGAgent(num_agents, obs_dim, action_dim, device)
    elif algo == "MAPPO":
        from mappo_agent import MAPPOAgent
        agent = MAPPOAgent(num_agents, obs_dim, action_dim, device)
    elif algo == "IDDPG":
        from iddpg_agent import IDDPGAgent
        agent = IDDPGAgent(num_agents, obs_dim, action_dim, device)
    else:
        raise ValueError(f"未知算法: {algo}")
    
    return agent


def test(algo, model_name=None):
    """测试指定算法的模型"""
    # 临时设置算法类型
    original_algo = Config.ALGORITHM
    Config.ALGORITHM = algo
    
    # 路径配置
    model_dir = Config.get_model_dir()
    
    # 自动查找模型
    if model_name is None:
        # 优先查找best模型
        model_name_base = Config.get_model_name()
        best_path = f"{model_dir}/{model_name_base}_best.pth"
        if os.path.exists(best_path):
            model_path = best_path
        else:
            model_path = f"{model_dir}/{model_name_base}.pth"
    else:
        model_path = f"{model_dir}/{model_name}"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print(f"   请先训练 {algo} 模型")
        Config.ALGORITHM = original_algo
        return
    
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    obstacle_mode = "随机" if Config.USE_RANDOM_OBSTACLES else "固定"
    
    print(f"\n🎯 测试配置 [{algo}]:")
    print(f"   障碍物: {obstacle_mode}")
    print(f"   起点: {start_mode}")
    print(f"   终点: {target_mode}")
    print(f"📁 加载模型: {model_path}")
    
    # 创建环境
    env = MultiAgentDroneEnv()
    obs_list = env.reset()
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    print(f"🤖 智能体数量: {num_agents}")
    
    # 加载agent
    agent = load_agent(algo, num_agents, obs_dim, action_dim, device)
    agent.load_model(model_path)
    print(f"✅ 模型加载成功!\n")
    
    # 测试统计
    test_episodes = 50
    total_success = 0
    total_collision = 0
    all_success_count = 0
    total_rewards = []
    total_steps = []
    
    print(f"{'='*60}")
    print(f"开始测试 - 共 {test_episodes} 轮")
    print(f"{'='*60}\n")
    
    for episode in range(1, test_episodes + 1):
        obs_list = env.reset()
        episode_rewards = [0.0] * num_agents
        episode_step = 0
        done = False
        
        while not done:
            # 选择动作（评估模式）
            if algo == "MAPPO":
                actions, _, _ = agent.select_actions(obs_list, eval_mode=True)
            else:
                actions = agent.select_actions(obs_list, eval_mode=True)
            
            next_obs_list, rewards, dones, info = env.step(actions)
            
            for i in range(num_agents):
                episode_rewards[i] += rewards[i]
            
            obs_list = next_obs_list
            episode_step += 1
            done = all(dones) or info['all_done']
            
            if Config.RENDER_TEST:
                env.render()
        
        # 统计
        total_reward = sum(episode_rewards)
        num_success = info['num_success']
        num_collision = info['num_collision']
        is_all_success = info['all_success']
        
        total_success += num_success
        total_collision += num_collision
        if is_all_success:
            all_success_count += 1
        total_rewards.append(total_reward)
        total_steps.append(episode_step)
        
        status = "🏆 ALL SUCCESS" if is_all_success else f"Success: {num_success}/{num_agents}"
        print(f"Episode {episode:3d} | Reward: {total_reward:7.1f} | "
              f"Steps: {episode_step:3d} | {status}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"{algo} 测试结果:")
    print(f"{'='*60}")
    print(f"总测试轮数: {test_episodes}")
    print(f"总成功次数: {total_success} / {test_episodes * num_agents} "
          f"({total_success / (test_episodes * num_agents) * 100:.1f}%)")
    print(f"所有Agent都成功: {all_success_count} / {test_episodes} "
          f"({all_success_count / test_episodes * 100:.1f}%)")
    print(f"总碰撞次数: {total_collision}")
    print(f"平均总奖励: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"平均步数: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"{'='*60}")
    
    env.close()
    Config.ALGORITHM = original_algo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试多智能体算法')
    parser.add_argument('--algo', type=str, required=True, 
                       choices=['MADDPG', 'MAPPO', 'IDDPG'],
                       help='算法类型')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件名（可选，默认自动查找）')
    
    args = parser.parse_args()
    
    test(args.algo, args.model)

