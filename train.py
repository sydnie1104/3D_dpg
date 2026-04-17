# train.py - MADDPG多智能体训练脚本
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

import os
import time
import csv
import numpy as np
from collections import deque
from env import MultiAgentDroneEnv
from agent import MADDPGAgent
from config import Config


def init_csv_file(file_path):
    """初始化CSV文件"""
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头：episode, 总奖励, 各agent奖励, 成功数, 碰撞数, 步数
        headers = ['episode', 'total_reward', 'avg_reward']
        for i in range(Config.NUM_AGENTS):
            headers.append(f'agent_{i}_reward')
        headers.extend(['num_success', 'num_collision', 'all_success', 'steps', 'noise_scale'])
        writer.writerow(headers)


def append_to_csv(file_path, data):
    """追加数据到CSV"""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def train():
    """主训练函数"""
    # 打印配置
    Config.print_obs_dims()
    
    # 路径配置
    model_dir = Config.get_model_dir()
    log_dir = Config.get_log_dir()
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = f"{model_dir}/{Config.MODEL_NAME}.pth"
    best_model_path = f"{model_dir}/{Config.MODEL_NAME}_best.pth"
    csv_log_path = f"{log_dir}/training_metrics.csv"
    
    obstacle_mode = "随机" if Config.USE_RANDOM_OBSTACLES else "固定"
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    
    print(f"\n🎯 训练配置:")
    print(f"   障碍物: {obstacle_mode}")
    print(f"   起点: {start_mode}")
    print(f"   终点: {target_mode}")
    print(f"📁 模型保存路径: {model_dir}")
    print(f"📊 日志保存路径: {log_dir}")
    print(f"🤖 智能体数量: {Config.NUM_AGENTS}")
    
    # 初始化CSV
    if not os.path.exists(csv_log_path):
        init_csv_file(csv_log_path)
    
    # 创建环境和智能体
    env = MultiAgentDroneEnv()
    obs_list = env.reset()
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    print(f"\n📐 观测维度: {obs_dim}")
    print(f"📐 动作维度: {action_dim}")
    print(f"📐 实际观测维度: {len(obs_list[0])}")
    
    agent = MADDPGAgent(num_agents, obs_dim, action_dim, device)
    
    # 训练统计
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=100)
    total_success = 0
    total_collision = 0
    all_success_count = 0  # 所有agent都成功的次数
    
    print(f"\n{'='*60}")
    print(f"开始训练 - 共 {Config.TOTAL_EPISODES} 轮")
    print(f"{'='*60}\n")
    
    try:
        for episode in range(1, Config.TOTAL_EPISODES + 1):
            obs_list = env.reset()
            agent.reset_noise()
            
            episode_rewards = [0.0] * num_agents
            episode_step = 0
            done = False
            
            while not done:
                # 选择动作
                actions = agent.select_actions(obs_list)
                
                # 执行动作
                next_obs_list, rewards, dones, info = env.step(actions)
                
                # 累计奖励
                for i in range(num_agents):
                    episode_rewards[i] += rewards[i]
                
                # 存储经验
                agent.replay_buffer.push(obs_list, actions, rewards, next_obs_list, dones)
                
                # 更新网络（每4步更新一次）
                if episode_step % 4 == 0 and len(agent.replay_buffer) >= Config.BATCH_SIZE:
                    agent.update()
                
                obs_list = next_obs_list
                episode_step += 1
                
                # 检查是否全部完成
                done = all(dones) or info['all_done']
            
            # Episode结束统计
            total_reward = sum(episode_rewards)
            avg_reward = total_reward / num_agents
            num_success = info['num_success']
            num_collision = info['num_collision']
            is_all_success = info['all_success']
            
            total_success += num_success
            total_collision += num_collision
            if is_all_success:
                all_success_count += 1
            
            # 噪声衰减
            agent.decay_noise()
            
            recent_rewards.append(total_reward)
            recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0
            
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
                episode_step,
                round(agent.noise_scale, 4)
            ])
            append_to_csv(csv_log_path, csv_data)
            
            # 打印进度
            if episode % 10 == 0 or is_all_success:
                success_rate = total_success / (episode * num_agents) * 100
                collision_rate = total_collision / (episode * num_agents) * 100
                all_success_rate = all_success_count / episode * 100
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.1f} | "
                      f"Avg: {avg_reward:6.1f} | "
                      f"Success: {num_success}/{num_agents} | "
                      f"Steps: {episode_step:3d} | "
                      f"AllSuccess: {all_success_rate:5.1f}% | "
                      f"Noise: {agent.noise_scale:.3f}")
            
            # 保存最佳模型
            if total_reward > best_reward and episode > Config.TOTAL_EPISODES * 0.1:
                best_reward = total_reward
                agent.save_model(best_model_path)
                print(f"  ✅ 保存最佳模型 (reward: {total_reward:.1f})")
            
            # 定期保存
            if episode % 100 == 0:
                agent.save_model(f"{model_dir}/{Config.MODEL_NAME}_ep{episode}.pth")
            
            # 所有agent都成功时保存
            if is_all_success:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                success_path = f"{model_dir}/{Config.MODEL_NAME}_allsuccess_ep{episode}_{timestamp}.pth"
                agent.save_model(success_path)
                print(f"  🏆 所有Agent成功! 保存模型: {success_path}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被中断")
    
    finally:
        # 保存最终模型
        agent.save_model(model_path)
        print(f"\n💾 最终模型已保存: {model_path}")
        
        # 打印最终统计
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总Episode: {episode}")
        print(f"总成功次数: {total_success}")
        print(f"总碰撞次数: {total_collision}")
        print(f"所有Agent都成功的次数: {all_success_count}")
        print(f"最佳总奖励: {best_reward:.1f}")
        print(f"{'='*60}")
        
        env.close()


if __name__ == '__main__':
    train()
