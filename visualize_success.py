# visualize_success.py - 多智能体成功轨迹可视化
"""
可视化多智能体成功的episode
从4个不同角度展示所有agent的轨迹
支持多算法：MADDPG, MAPPO, IDDPG
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import csv
import argparse
from copy import deepcopy

from env import MultiAgentDroneEnv
from config import Config


# Agent颜色
AGENT_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']


def draw_box(ax, center, half_width, half_length, height, color='#7f7f7f', alpha=0.8):
    """绘制长方体障碍物"""
    cx, cy = center
    x_min, x_max = cx - half_width, cx + half_width
    y_min, y_max = cy - half_length, cy + half_length
    z_min, z_max = 0, height
    
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]
    
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidths=0.3)
    ax.add_collection3d(poly)


def draw_sphere(ax, center, radius, color='orange', alpha=0.5, resolution=15):
    """绘制球体"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def visualize_multi_agent_trajectory(path_histories, start_positions, target_positions, 
                                     obstacles, dynamic_obstacles_final, 
                                     grid_scale, save_prefix='success'):
    """
    可视化多智能体轨迹
    
    Args:
        path_histories: List[List[Point]], 每个agent的路径历史
        start_positions: List[np.array], 每个agent的起点
        target_positions: List[np.array], 每个agent的终点
        obstacles: 静态障碍物列表
        dynamic_obstacles_final: 动态障碍物最终位置列表
        grid_scale: 网格大小
        save_prefix: 保存文件前缀
    """
    num_agents = len(path_histories)
    fig = plt.figure(figsize=(20, 15))
    
    # 4个视角
    views = [
        {'elev': 30, 'azim': -60, 'title': '训练视角 (Training View)'},
        {'elev': 0, 'azim': 0, 'title': '正视图 (Front View)'},
        {'elev': 90, 'azim': 0, 'title': '俯视图 (Top View)'},
        {'elev': 20, 'azim': 135, 'title': '后斜视图 (Back Perspective)'}
    ]
    
    for idx, view in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 1. 绘制静态障碍物
        for obs in obstacles:
            ox, oy, bd_w, bd_l, h = obs
            draw_box(ax, (ox, oy), bd_w, bd_l, h, alpha=0.7)
        
        # 2. 绘制动态障碍物（最终位置）
        if dynamic_obstacles_final:
            for dyn_obs in dynamic_obstacles_final:
                if hasattr(dyn_obs, 'pos'):
                    pos = dyn_obs.pos.to_array()
                    radius = dyn_obs.radius
                else:
                    pos = dyn_obs['pos']
                    radius = dyn_obs['radius']
                draw_sphere(ax, pos, radius, color='orange', alpha=0.4)
        
        # 3. 绘制每个agent的轨迹、起点、终点
        for i in range(num_agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            
            # 轨迹
            if path_histories[i]:
                trajectory = np.array([[p.x, p.y, p.z] for p in path_histories[i]])
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                       color=color, linewidth=2, label=f'Agent {i}', zorder=10)
            
            # 起点
            ax.scatter(*start_positions[i], c=color, s=180, marker='o',
                      edgecolors='black', linewidths=2, zorder=20)
            
            # 终点
            ax.scatter(*target_positions[i], c=color, s=280, marker='*',
                      edgecolors='black', linewidths=2, zorder=20)
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
        ax.set_xlim(0, grid_scale)
        ax.set_ylim(0, grid_scale)
        ax.set_zlim(0, grid_scale)
        
        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_title(view['title'], fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('visualizations', exist_ok=True)
    save_path = f'visualizations/{save_prefix}_4views.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" 可视化已保存: {save_path}")
    
    # 保存轨迹数据
    csv_path = f'visualizations/{save_prefix}_trajectories.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step']
        for i in range(num_agents):
            header.extend([f'agent_{i}_x', f'agent_{i}_y', f'agent_{i}_z'])
        writer.writerow(header)
        
        max_steps = max(len(ph) for ph in path_histories)
        for step in range(max_steps):
            row = [step]
            for i in range(num_agents):
                if step < len(path_histories[i]):
                    p = path_histories[i][step]
                    row.extend([p.x, p.y, p.z])
                else:
                    # 已完成的agent保持最后位置
                    p = path_histories[i][-1]
                    row.extend([p.x, p.y, p.z])
            writer.writerow(row)
    print(f" 轨迹数据已保存: {csv_path}")
    
    plt.show()


def run_single_episode(env, agent, algo="MADDPG"):
    """运行一个episode并返回结果"""
    obs_list = env.reset()
    if hasattr(agent, 'reset_noise'):
        agent.reset_noise()
    
    done = False
    total_reward = 0.0
    episode_rewards = [0.0] * env.num_agents
    
    while not done:
        # 根据算法类型选择动作
        if algo == "MAPPO":
            actions, _, _ = agent.select_actions(obs_list, eval_mode=True)
        else:
            actions = agent.select_actions(obs_list, eval_mode=True)
        next_obs_list, rewards, dones, info = env.step(actions)
        
        for i in range(env.num_agents):
            episode_rewards[i] += rewards[i]
        total_reward += sum(rewards)
        
        obs_list = next_obs_list
        done = all(dones) or info['all_done']
    
    # 收集结果
    result = {
        'all_success': info['all_success'],
        'num_success': info['num_success'],
        'num_collision': info['num_collision'],
        'total_reward': total_reward,
        'episode_rewards': episode_rewards,
        'step_count': env.step_count,
        'path_histories': [deepcopy(ph) for ph in env.path_histories],
        'start_positions': [env.path_histories[i][0].to_array() for i in range(env.num_agents)],
        'target_positions': [tp.to_array() for tp in env.target_positions],
        'obstacles': deepcopy(env.obstacles),
        'dynamic_obstacles': deepcopy(env.dynamic_obstacles),
        'agent_success': env.agent_success.copy(),
        'agent_collided': env.agent_collided.copy()
    }
    
    return result


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


def main(algo="MADDPG"):
    """主函数"""
    # 临时设置算法类型
    original_algo = Config.ALGORITHM
    Config.ALGORITHM = algo
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    print(f" 算法: {algo}")
    
    # 打印配置
    Config.print_obs_dims()
    
    # 显示起点终点模式
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    obstacle_mode = "随机" if Config.USE_RANDOM_OBSTACLES else "固定"
    print(f"\n 起点模式: {start_mode}")
    print(f" 终点模式: {target_mode}")
    print(f" 障碍物模式: {obstacle_mode}")
    
    # 创建环境
    print("\n 创建多智能体环境...")
    env = MultiAgentDroneEnv()
    
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    print(f"   Agent数量: {num_agents}")
    print(f"   观测维度: {obs_dim}")
    print(f"   动作维度: {action_dim}")
    
    # 创建智能体
    print(f"\n 创建{algo}智能体...")
    agent = load_agent(algo, num_agents, obs_dim, action_dim, device)
    
    # 设置为评估模式
    if hasattr(agent, 'actors'):
        for actor in agent.actors:
            actor.eval()
    if hasattr(agent, 'critics'):
        for critic in agent.critics:
            critic.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()
    
    # 查找模型
    print("\n🔍 查找模型...")
    model_dir = Config.get_model_dir()
    
    # 优先查找全部成功的模型
    if not os.path.exists(model_dir):
        print(f"    模型目录不存在: {model_dir}")
        print(f"   请先训练 {algo} 模型")
        Config.ALGORITHM = original_algo
        return
    
    success_models = [f for f in os.listdir(model_dir) 
                     if 'allsuccess' in f.lower() and f.endswith('.pth')]
    
    if success_models:
        success_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        model_path = os.path.join(model_dir, success_models[0])
        print(f"   使用全部成功模型: {success_models[0]}")
    else:
        # 查找最佳模型或最新模型
        model_name = Config.get_model_name()
        best_path = f"{model_dir}/{model_name}_best.pth"
        if os.path.exists(best_path):
            model_path = best_path
            print(f"   使用最佳模型: {model_name}_best.pth")
        else:
            all_models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if not all_models:
                print("    未找到任何模型!")
                Config.ALGORITHM = original_algo
                return
            all_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            model_path = os.path.join(model_dir, all_models[0])
            print(f"   使用最新模型: {all_models[0]}")
    
    # 加载模型
    print(f"\n 加载模型: {model_path}")
    try:
        agent.load_model(model_path)
        print("    模型加载成功!")
    except Exception as e:
        print(f"    模型加载失败: {e}")
        return
    
    # 运行测试找全部成功的episode
    target_visualizations = 10
    success_count = 0
    attempt = 0
    max_attempts = 200
    
    print(f"\n{'='*60}")
    print(f" 目标: 找到 {target_visualizations} 个全部成功的episode并可视化")
    print(f"{'='*60}\n")
    
    while success_count < target_visualizations and attempt < max_attempts:
        attempt += 1
        result = run_single_episode(env, agent, algo)
        
        status = "" if result['all_success'] else f"{result['num_success']}/{num_agents}"
        print(f"尝试 {attempt:3d}: {status} | "
              f"Steps={result['step_count']:3d} | "
              f"Reward={result['total_reward']:.1f}")
        
        if result['all_success']:
            success_count += 1
            save_prefix = f"multi_success_{success_count:02d}_ep{attempt:03d}"
            
            print(f"  → 全部成功! 生成可视化...")
            visualize_multi_agent_trajectory(
                path_histories=result['path_histories'],
                start_positions=result['start_positions'],
                target_positions=result['target_positions'],
                obstacles=result['obstacles'],
                dynamic_obstacles_final=result['dynamic_obstacles'],
                grid_scale=env.grid_scale,
                save_prefix=save_prefix
            )
    
    if success_count < target_visualizations:
        print(f"\n 只找到 {success_count}/{target_visualizations} 个全部成功的episode")
    else:
        print(f"\n 已保存 {target_visualizations} 个全部成功案例的可视化!")
    
    env.close()
    Config.ALGORITHM = original_algo  # 恢复原设置


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化多智能体成功轨迹')
    parser.add_argument('--algo', type=str, default='MADDPG',
                       choices=['MADDPG', 'MAPPO', 'IDDPG'],
                       help='算法类型（默认: MADDPG）')
    args = parser.parse_args()
    
    main(args.algo)
