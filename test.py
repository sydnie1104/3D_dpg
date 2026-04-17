# test.py - MADDPG多智能体测试脚本
import torch
import os
import csv
import numpy as np
from env import MultiAgentDroneEnv
from agent import MADDPGAgent
from config import Config


def init_test_csv(path):
    """初始化测试CSV"""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['episode', 'total_reward', 'avg_reward']
        for i in range(Config.NUM_AGENTS):
            headers.append(f'agent_{i}_reward')
        headers.extend(['num_success', 'num_collision', 'all_success', 'steps'])
        writer.writerow(headers)


def append_test_csv(path, data):
    """追加测试数据"""
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def test():
    """测试函数"""
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on device: {device}")
    
    # 打印配置
    Config.print_obs_dims()
    
    # 显示起点终点模式
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    print(f"\n📍 起点模式: {start_mode}")
    print(f"🎯 终点模式: {target_mode}")
    
    # 路径
    model_dir = Config.get_model_dir()
    log_dir = Config.get_log_dir()
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    test_log_path = f"{log_dir}/test_metrics.csv"
    
    # 查找最佳模型
    model_path = f"{model_dir}/{Config.MODEL_NAME}_best.pth"
    if not os.path.exists(model_path):
        # 查找其他模型
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            print(f"❌ 未找到模型文件: {model_dir}")
            return
        # 选择最新的模型
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        model_path = os.path.join(model_dir, model_files[0])
    
    print(f"\n📥 加载模型: {model_path}")
    
    # 初始化CSV
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    init_test_csv(test_log_path)
    
    # 创建环境和智能体
    env = MultiAgentDroneEnv()
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    agent = MADDPGAgent(num_agents, obs_dim, action_dim, device)
    
    try:
        agent.load_model(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试配置
    test_episodes = getattr(Config, 'TEST_EPISODES', 100)
    
    # 统计
    total_success = 0
    total_collision = 0
    all_success_count = 0
    all_rewards = []
    
    print(f"\n{'='*60}")
    print(f"开始测试 - 共 {test_episodes} 轮")
    print(f"{'='*60}\n")
    
    for episode in range(1, test_episodes + 1):
        obs_list = env.reset()
        
        episode_rewards = [0.0] * num_agents
        episode_step = 0
        done = False
        
        while not done:
            # 评估模式选择动作（无噪声）
            actions = agent.select_actions(obs_list, eval_mode=True)
            
            # 执行动作
            next_obs_list, rewards, dones, info = env.step(actions)
            
            # 累计奖励
            for i in range(num_agents):
                episode_rewards[i] += rewards[i]
            
            obs_list = next_obs_list
            episode_step += 1
            
            done = all(dones) or info['all_done']
        
        # Episode统计
        total_reward = sum(episode_rewards)
        avg_reward = total_reward / num_agents
        num_success = info['num_success']
        num_collision = info['num_collision']
        is_all_success = info['all_success']
        
        total_success += num_success
        total_collision += num_collision
        if is_all_success:
            all_success_count += 1
        all_rewards.append(total_reward)
        
        # 记录到CSV
        csv_data = [
            episode,
            round(total_reward, 2),
            round(avg_reward, 2)
        ]
        for i in range(num_agents):
            csv_data.append(round(episode_rewards[i], 2))
        csv_data.extend([
            num_success,
            num_collision,
            1 if is_all_success else 0,
            episode_step
        ])
        append_test_csv(test_log_path, csv_data)
        
        # 打印进度
        status = "🏆" if is_all_success else ("✅" if num_success > 0 else "❌")
        print(f"Episode {episode:3d} | "
              f"Reward: {total_reward:7.1f} | "
              f"Success: {num_success}/{num_agents} | "
              f"Steps: {episode_step:3d} | {status}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")
    print(f"总测试轮次: {test_episodes}")
    print(f"Agent成功率: {total_success}/{test_episodes * num_agents} = {total_success/(test_episodes*num_agents)*100:.2f}%")
    print(f"Agent碰撞率: {total_collision}/{test_episodes * num_agents} = {total_collision/(test_episodes*num_agents)*100:.2f}%")
    print(f"全部成功率: {all_success_count}/{test_episodes} = {all_success_count/test_episodes*100:.2f}%")
    print(f"平均总奖励: {np.mean(all_rewards):.2f}")
    print(f"最高总奖励: {np.max(all_rewards):.2f}")
    print(f"最低总奖励: {np.min(all_rewards):.2f}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == '__main__':
    test()
