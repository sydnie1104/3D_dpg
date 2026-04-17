# test_compare_all.py - 三算法对比测试脚本
"""
测试 MADDPG, IDDPG, MAPPO 三种算法，对比：
- 总奖励
- 成功率
- 路径长度（仅成功轮次）
"""
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import MultiAgentDroneEnv
from config import Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_agent(algo, num_agents, obs_dim, action_dim, device):
    """根据算法类型加载agent"""
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


def find_model_path(algo):
    """查找指定算法的最佳模型"""
    # MADDPG使用原始models目录
    if algo == "MADDPG":
        model_dir = "models"
        model_name = "maddpg_model"
    else:
        original_algo = Config.ALGORITHM
        Config.ALGORITHM = algo
        model_dir = Config.get_model_dir()
        model_name = Config.get_model_name()
        Config.ALGORITHM = original_algo
    
    # 优先查找best模型
    best_path = f"{model_dir}/{model_name}_best.pth"
    if os.path.exists(best_path):
        return best_path
    
    # 否则查找最新模型
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])


def test_algorithm(algo, env, num_agents, obs_dim, action_dim, device, test_episodes=100):
    """测试单个算法"""
    print(f"\n{'='*60}")
    print(f"🧪 测试 {algo}")
    print(f"{'='*60}")
    
    # 查找模型
    model_path = find_model_path(algo)
    if model_path is None or not os.path.exists(model_path):
        print(f"❌ 未找到 {algo} 模型")
        return None
    
    print(f"📥 加载模型: {model_path}")
    
    # 加载agent
    agent = load_agent(algo, num_agents, obs_dim, action_dim, device)
    
    try:
        agent.load_model(model_path)
        print(f"✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 测试统计
    results = {
        'algo': algo,
        'episodes': [],
        'total_rewards': [],
        'agent_rewards': [[] for _ in range(num_agents)],
        'num_success': [],
        'num_collision': [],
        'all_success': [],
        'steps': [],
        'path_lengths': []  # 每个episode的所有agent路径长度
    }
    
    print(f"开始测试 {test_episodes} 轮...")
    
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
        
        # 统计
        total_reward = sum(episode_rewards)
        num_success = info['num_success']
        num_collision = info['num_collision']
        is_all_success = info['all_success']
        
        # 计算路径长度
        path_lengths = []
        for i in range(num_agents):
            if len(env.path_histories[i]) > 1:
                path_length = 0.0
                for j in range(1, len(env.path_histories[i])):
                    prev = env.path_histories[i][j-1].to_array()
                    curr = env.path_histories[i][j].to_array()
                    path_length += np.linalg.norm(curr - prev)
                path_lengths.append(path_length)
            else:
                path_lengths.append(0.0)
        
        # 记录结果
        results['episodes'].append(episode)
        results['total_rewards'].append(total_reward)
        for i in range(num_agents):
            results['agent_rewards'][i].append(episode_rewards[i])
        results['num_success'].append(num_success)
        results['num_collision'].append(num_collision)
        results['all_success'].append(1 if is_all_success else 0)
        results['steps'].append(episode_step)
        results['path_lengths'].append(path_lengths)
        
        # 打印进度
        if episode % 20 == 0 or is_all_success:
            status = "🏆" if is_all_success else f"✅{num_success}/{num_agents}"
            print(f"  Episode {episode:3d}/{test_episodes} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"{status}")
    
    print(f"✅ {algo} 测试完成!")
    return results


def calculate_statistics(results):
    """计算统计数据"""
    num_agents = Config.NUM_AGENTS
    test_episodes = len(results['episodes'])
    
    stats = {
        'algo': results['algo'],
        'test_episodes': test_episodes,
        
        # 奖励统计
        'avg_total_reward': np.mean(results['total_rewards']),
        'std_total_reward': np.std(results['total_rewards']),
        'max_total_reward': np.max(results['total_rewards']),
        'min_total_reward': np.min(results['total_rewards']),
        
        # 成功率统计
        'total_success': sum(results['num_success']),
        'agent_success_rate': sum(results['num_success']) / (test_episodes * num_agents) * 100,
        'all_success_count': sum(results['all_success']),
        'all_success_rate': sum(results['all_success']) / test_episodes * 100,
        
        # 碰撞统计
        'total_collision': sum(results['num_collision']),
        'collision_rate': sum(results['num_collision']) / (test_episodes * num_agents) * 100,
        
        # 步数统计
        'avg_steps': np.mean(results['steps']),
        'std_steps': np.std(results['steps']),
    }
    
    # 路径长度统计（仅成功的agent）
    all_successful_paths = []
    for i, episode_paths in enumerate(results['path_lengths']):
        num_success = results['num_success'][i]
        if num_success > 0:
            # 按路径长度排序，取前num_success个（认为是成功的）
            sorted_paths = sorted([(j, p) for j, p in enumerate(episode_paths)], key=lambda x: x[1])
            for j in range(min(num_success, len(sorted_paths))):
                if sorted_paths[j][1] > 0:
                    all_successful_paths.append(sorted_paths[j][1])
    
    if len(all_successful_paths) > 0:
        stats['avg_success_path_length'] = np.mean(all_successful_paths)
        stats['std_success_path_length'] = np.std(all_successful_paths)
        stats['min_success_path_length'] = np.min(all_successful_paths)
        stats['max_success_path_length'] = np.max(all_successful_paths)
    else:
        stats['avg_success_path_length'] = 0
        stats['std_success_path_length'] = 0
        stats['min_success_path_length'] = 0
        stats['max_success_path_length'] = 0
    
    return stats


def print_comparison_table(all_stats):
    """打印对比表格"""
    print(f"\n{'='*100}")
    print("📊 三算法对比测试结果")
    print(f"{'='*100}\n")
    
    # 表头
    print(f"{'指标':<20} {'MADDPG':>15} {'IDDPG':>15} {'MAPPO':>15} {'最佳':>15}")
    print("-"*100)
    
    # 提取数据
    maddpg_stats = next((s for s in all_stats if s['algo'] == 'MADDPG'), None)
    iddpg_stats = next((s for s in all_stats if s['algo'] == 'IDDPG'), None)
    mappo_stats = next((s for s in all_stats if s['algo'] == 'MAPPO'), None)
    
    def format_value(val, fmt=".2f"):
        return f"{val:{fmt}}" if val is not None else "N/A"
    
    def get_best(values, higher_better=True):
        valid = [v for v in values if v is not None]
        if not valid:
            return "N/A"
        best_val = max(valid) if higher_better else min(valid)
        best_idx = values.index(best_val)
        algos = ['MADDPG', 'IDDPG', 'MAPPO']
        return algos[best_idx]
    
    # 奖励对比
    print("\n【奖励对比】")
    avg_rewards = [
        maddpg_stats['avg_total_reward'] if maddpg_stats else None,
        iddpg_stats['avg_total_reward'] if iddpg_stats else None,
        mappo_stats['avg_total_reward'] if mappo_stats else None
    ]
    print(f"{'平均总奖励':<20} {format_value(avg_rewards[0], '.2f'):>15} "
          f"{format_value(avg_rewards[1], '.2f'):>15} {format_value(avg_rewards[2], '.2f'):>15} "
          f"{get_best(avg_rewards):>15}")
    
    max_rewards = [
        maddpg_stats['max_total_reward'] if maddpg_stats else None,
        iddpg_stats['max_total_reward'] if iddpg_stats else None,
        mappo_stats['max_total_reward'] if mappo_stats else None
    ]
    print(f"{'最高总奖励':<20} {format_value(max_rewards[0], '.2f'):>15} "
          f"{format_value(max_rewards[1], '.2f'):>15} {format_value(max_rewards[2], '.2f'):>15} "
          f"{get_best(max_rewards):>15}")
    
    # 成功率对比
    print("\n【成功率对比】")
    agent_success = [
        maddpg_stats['agent_success_rate'] if maddpg_stats else None,
        iddpg_stats['agent_success_rate'] if iddpg_stats else None,
        mappo_stats['agent_success_rate'] if mappo_stats else None
    ]
    print(f"{'Agent成功率 (%)':<20} {format_value(agent_success[0], '.2f'):>15} "
          f"{format_value(agent_success[1], '.2f'):>15} {format_value(agent_success[2], '.2f'):>15} "
          f"{get_best(agent_success):>15}")
    
    all_success = [
        maddpg_stats['all_success_rate'] if maddpg_stats else None,
        iddpg_stats['all_success_rate'] if iddpg_stats else None,
        mappo_stats['all_success_rate'] if mappo_stats else None
    ]
    print(f"{'全部成功率 (%)':<20} {format_value(all_success[0], '.2f'):>15} "
          f"{format_value(all_success[1], '.2f'):>15} {format_value(all_success[2], '.2f'):>15} "
          f"{get_best(all_success):>15}")
    
    # 碰撞率对比（越低越好）
    print("\n【碰撞率对比】")
    collision = [
        maddpg_stats['collision_rate'] if maddpg_stats else None,
        iddpg_stats['collision_rate'] if iddpg_stats else None,
        mappo_stats['collision_rate'] if mappo_stats else None
    ]
    print(f"{'碰撞率 (%)':<20} {format_value(collision[0], '.2f'):>15} "
          f"{format_value(collision[1], '.2f'):>15} {format_value(collision[2], '.2f'):>15} "
          f"{get_best(collision, False):>15}")
    
    # 路径长度对比（越短越好，仅成功轮次）
    print("\n【路径长度对比（仅成功）】")
    path_len = [
        maddpg_stats['avg_success_path_length'] if maddpg_stats else None,
        iddpg_stats['avg_success_path_length'] if iddpg_stats else None,
        mappo_stats['avg_success_path_length'] if mappo_stats else None
    ]
    print(f"{'平均路径长度':<20} {format_value(path_len[0], '.2f'):>15} "
          f"{format_value(path_len[1], '.2f'):>15} {format_value(path_len[2], '.2f'):>15} "
          f"{get_best(path_len, False):>15}")
    
    # 步数对比（越少越好）
    print("\n【步数对比】")
    steps = [
        maddpg_stats['avg_steps'] if maddpg_stats else None,
        iddpg_stats['avg_steps'] if iddpg_stats else None,
        mappo_stats['avg_steps'] if mappo_stats else None
    ]
    print(f"{'平均步数':<20} {format_value(steps[0], '.1f'):>15} "
          f"{format_value(steps[1], '.1f'):>15} {format_value(steps[2], '.1f'):>15} "
          f"{get_best(steps, False):>15}")
    
    print("\n" + "="*100)


def plot_comparison(all_stats, save_dir='visualizations'):
    """绘制对比图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    algos = [s['algo'] for s in all_stats]
    colors = {'MADDPG': '#1f77b4', 'IDDPG': '#ff7f0e', 'MAPPO': '#2ca02c'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('三算法测试对比 (100轮)', fontsize=16, fontweight='bold')
    
    # 1. 平均奖励对比
    ax1 = axes[0, 0]
    avg_rewards = [s['avg_total_reward'] for s in all_stats]
    std_rewards = [s['std_total_reward'] for s in all_stats]
    bars = ax1.bar(algos, avg_rewards, yerr=std_rewards, 
                   color=[colors[a] for a in algos], alpha=0.8, capsize=5)
    ax1.set_ylabel('平均总奖励', fontsize=11, fontweight='bold')
    ax1.set_title('平均总奖励对比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 成功率对比
    ax2 = axes[0, 1]
    agent_success = [s['agent_success_rate'] for s in all_stats]
    all_success = [s['all_success_rate'] for s in all_stats]
    x = np.arange(len(algos))
    width = 0.35
    ax2.bar(x - width/2, agent_success, width, label='Agent成功率', alpha=0.8)
    ax2.bar(x + width/2, all_success, width, label='全部成功率', alpha=0.8)
    ax2.set_ylabel('成功率 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('成功率对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    # 3. 碰撞率对比
    ax3 = axes[0, 2]
    collision_rates = [s['collision_rate'] for s in all_stats]
    bars = ax3.bar(algos, collision_rates, 
                   color=[colors[a] for a in algos], alpha=0.8)
    ax3.set_ylabel('碰撞率 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('碰撞率对比（越低越好）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, collision_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 路径长度对比
    ax4 = axes[1, 0]
    path_lengths = [s['avg_success_path_length'] for s in all_stats]
    path_stds = [s['std_success_path_length'] for s in all_stats]
    bars = ax4.bar(algos, path_lengths, yerr=path_stds,
                   color=[colors[a] for a in algos], alpha=0.8, capsize=5)
    ax4.set_ylabel('平均路径长度', fontsize=11, fontweight='bold')
    ax4.set_title('成功路径长度对比（越短越好）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, path_lengths):
        ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. 步数对比
    ax5 = axes[1, 1]
    avg_steps = [s['avg_steps'] for s in all_stats]
    std_steps = [s['std_steps'] for s in all_stats]
    bars = ax5.bar(algos, avg_steps, yerr=std_steps,
                   color=[colors[a] for a in algos], alpha=0.8, capsize=5)
    ax5.set_ylabel('平均步数', fontsize=11, fontweight='bold')
    ax5.set_title('平均步数对比（越少越好）', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_steps):
        ax5.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. 综合得分雷达图
    ax6 = axes[1, 2]
    categories = ['奖励', '成功率', '全部成功', '低碰撞', '短路径']
    
    # 归一化指标（0-1之间，越高越好）
    def normalize_scores(all_stats):
        scores = {}
        for s in all_stats:
            algo = s['algo']
            scores[algo] = [
                s['avg_total_reward'] / max([st['avg_total_reward'] for st in all_stats]),  # 奖励
                s['agent_success_rate'] / 100,  # 成功率
                s['all_success_rate'] / 100,  # 全部成功
                1 - (s['collision_rate'] / 100),  # 低碰撞（反转）
                1 - (s['avg_success_path_length'] / max([st['avg_success_path_length'] for st in all_stats if st['avg_success_path_length'] > 0]))  # 短路径（反转）
            ]
        return scores
    
    scores = normalize_scores(all_stats)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    for algo in algos:
        values = scores[algo] + scores[algo][:1]
        ax6.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[algo])
        ax6.fill(angles, values, alpha=0.15, color=colors[algo])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.set_title('综合性能雷达图', fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.grid(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'test_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")
    
    # 配置
    Config.print_obs_dims()
    
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    obstacle_mode = "随机" if Config.USE_RANDOM_OBSTACLES else "固定"
    
    print(f"\n📍 测试配置:")
    print(f"   起点: {start_mode}")
    print(f"   终点: {target_mode}")
    print(f"   障碍物: {obstacle_mode}")
    print(f"   Agent数量: {Config.NUM_AGENTS}")
    
    # 创建环境
    env = MultiAgentDroneEnv()
    obs_dim = Config.get_single_obs_dim()
    action_dim = Config.get_action_dim()
    num_agents = Config.NUM_AGENTS
    
    test_episodes = 100
    
    # 测试三个算法
    algos = ["MADDPG", "IDDPG", "MAPPO"]
    all_results = []
    all_stats = []
    
    for algo in algos:
        results = test_algorithm(algo, env, num_agents, obs_dim, action_dim, device, test_episodes)
        if results is not None:
            all_results.append(results)
            stats = calculate_statistics(results)
            all_stats.append(stats)
    
    env.close()
    
    if len(all_stats) == 0:
        print("\n❌ 没有成功测试任何算法！")
        return
    
    # 打印对比表格
    print_comparison_table(all_stats)
    
    # 绘制对比图表
    plot_comparison(all_stats)
    
    print("\n🎉 所有测试完成!")


if __name__ == '__main__':
    main()

