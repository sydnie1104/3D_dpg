"""
多智能体训练指标可视化脚本
绘制核心性能指标：总奖励、成功率、全部成功率、碰撞率、步数等
支持MADDPG, MAPPO, IDDPG三种算法
支持单算法可视化和三算法对比
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from config import Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def moving_average(data, window_size=50):
    """计算移动平均"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def load_training_data(algo):
    """加载指定算法的训练数据"""
    # MADDPG使用原始logs目录，其他算法使用logs_{algo}目录
    if algo == "MADDPG":
        csv_path = 'logs/training_metrics.csv'
    else:
        original_algo = Config.ALGORITHM
        Config.ALGORITHM = algo
        csv_path = f'{Config.get_log_dir()}/training_metrics.csv'
        Config.ALGORITHM = original_algo
    
    if not os.path.exists(csv_path):
        print(f"⚠️  {algo} 数据不存在: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✅ {algo}: {len(df)} 轮训练数据")
    return df


def plot_multi_agent_training_metrics(algo="MADDPG", csv_path=None, save_dir='visualizations'):
    """绘制多智能体训练指标"""
    
    # 临时设置算法类型
    original_algo = Config.ALGORITHM
    Config.ALGORITHM = algo
    
    if csv_path is None:
        csv_path = f'{Config.get_log_dir()}/training_metrics.csv'
    
    print(f"📖 读取训练数据: {csv_path}")
    print(f"🎯 算法: {algo}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        Config.ALGORITHM = original_algo
        return
    
    df = pd.read_csv(csv_path)
    print(f"   总训练轮次: {len(df)}")
    print(f"   数据列: {list(df.columns)}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_agents = Config.NUM_AGENTS
    
    # 创建图表 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{algo}多智能体训练指标 ({num_agents} Agents)', fontsize=16, fontweight='bold', y=0.995)
    
    window_size = 50
    episodes = df['episode'].values
    
    # ==================== 1. 总奖励曲线 ====================
    ax1 = axes[0, 0]
    total_reward = df['total_reward'].values
    
    ax1.plot(episodes, total_reward, alpha=0.3, color='blue', linewidth=1, label='原始数据')
    if len(total_reward) >= window_size:
        ma_episodes = episodes[window_size-1:]
        ma_reward = moving_average(total_reward, window_size)
        ax1.plot(ma_episodes, ma_reward, color='blue', linewidth=2.5, label=f'{window_size}轮移动平均')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax1.set_ylabel('总奖励', fontsize=11, fontweight='bold')
    ax1.set_title('1.1 总奖励曲线', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9)
    
    # 统计信息
    final_reward = total_reward[-100:].mean() if len(total_reward) >= 100 else total_reward.mean()
    ax1.text(0.02, 0.98, f'最终100轮平均: {final_reward:.1f}\n最高: {total_reward.max():.1f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== 2. Agent成功率 ====================
    ax2 = axes[0, 1]
    success_rate = df['num_success'].values / num_agents * 100
    
    ax2.plot(episodes, success_rate, alpha=0.3, color='green', linewidth=1, label='原始数据')
    if len(success_rate) >= window_size:
        ma_success = moving_average(success_rate, window_size)
        ax2.plot(ma_episodes, ma_success, color='green', linewidth=2.5, label=f'{window_size}轮移动平均')
    
    ax2.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Agent成功率 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('1.2 Agent成功率', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim(0, 105)
    
    final_success = success_rate[-100:].mean() if len(success_rate) >= 100 else success_rate.mean()
    ax2.text(0.02, 0.98, f'最终100轮平均: {final_success:.1f}%',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== 3. 全部成功率 ====================
    ax3 = axes[0, 2]
    all_success = df['all_success'].values * 100
    
    # 累计全部成功率
    cumsum_success = np.cumsum(all_success / 100)
    all_success_rate = cumsum_success / np.arange(1, len(all_success) + 1) * 100
    
    ax3.plot(episodes, all_success_rate, color='darkgreen', linewidth=2, label='累计全部成功率')
    ax3.fill_between(episodes, 0, all_success_rate, alpha=0.3, color='green')
    
    ax3.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax3.set_ylabel('全部成功率 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('1.3 全部Agent成功率', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=9)
    ax3.set_ylim(0, 105)
    
    final_all_success = all_success_rate[-1] if len(all_success_rate) > 0 else 0
    ax3.text(0.02, 0.98, f'最终全部成功率: {final_all_success:.1f}%\n总成功次数: {int(cumsum_success[-1])}',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== 4. 碰撞率 ====================
    ax4 = axes[1, 0]
    collision_rate = df['num_collision'].values / num_agents * 100
    
    ax4.plot(episodes, collision_rate, alpha=0.3, color='red', linewidth=1, label='原始数据')
    if len(collision_rate) >= window_size:
        ma_collision = moving_average(collision_rate, window_size)
        ax4.plot(ma_episodes, ma_collision, color='red', linewidth=2.5, label=f'{window_size}轮移动平均')
    
    ax4.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax4.set_ylabel('碰撞率 (%)', fontsize=11, fontweight='bold')
    ax4.set_title('1.4 Agent碰撞率', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='best', fontsize=9)
    ax4.set_ylim(0, 105)
    
    final_collision = collision_rate[-100:].mean() if len(collision_rate) >= 100 else collision_rate.mean()
    ax4.text(0.02, 0.98, f'最终100轮平均: {final_collision:.1f}%',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== 5. 步数 ====================
    ax5 = axes[1, 1]
    steps = df['steps'].values
    
    ax5.plot(episodes, steps, alpha=0.3, color='purple', linewidth=1, label='原始数据')
    if len(steps) >= window_size:
        ma_steps = moving_average(steps, window_size)
        ax5.plot(ma_episodes, ma_steps, color='purple', linewidth=2.5, label=f'{window_size}轮移动平均')
    
    ax5.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax5.set_ylabel('步数', fontsize=11, fontweight='bold')
    ax5.set_title('1.5 每轮步数', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='best', fontsize=9)
    
    final_steps = steps[-100:].mean() if len(steps) >= 100 else steps.mean()
    ax5.text(0.02, 0.98, f'最终100轮平均: {final_steps:.1f}步',
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== 6. 各Agent奖励对比 ====================
    ax6 = axes[1, 2]
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    for i in range(num_agents):
        col_name = f'agent_{i}_reward'
        if col_name in df.columns:
            agent_reward = df[col_name].values
            if len(agent_reward) >= window_size:
                ma_agent_reward = moving_average(agent_reward, window_size)
                ax6.plot(ma_episodes, ma_agent_reward, color=colors[i % len(colors)], 
                        linewidth=1.5, label=f'Agent {i}', alpha=0.8)
    
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('训练轮次', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Agent奖励 (移动平均)', fontsize=11, fontweight='bold')
    ax6.set_title('1.6 各Agent奖励对比', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'training_core_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {save_path}")
    
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("📊 训练统计摘要")
    print("="*60)
    print(f"总训练轮次: {len(df)}")
    print(f"Agent数量: {num_agents}")
    print(f"\n【总奖励】")
    print(f"  最终100轮平均: {final_reward:.2f}")
    print(f"  最高: {total_reward.max():.2f}")
    print(f"  最低: {total_reward.min():.2f}")
    print(f"\n【Agent成功率】")
    print(f"  最终100轮平均: {final_success:.2f}%")
    print(f"\n【全部成功率】")
    print(f"  最终: {final_all_success:.2f}%")
    print(f"  总成功次数: {int(cumsum_success[-1])}")
    print(f"\n【碰撞率】")
    print(f"  最终100轮平均: {final_collision:.2f}%")
    print(f"\n【步数】")
    print(f"  最终100轮平均: {final_steps:.1f}步")
    print("="*60)
    
    Config.ALGORITHM = original_algo  # 恢复原设置


def plot_algorithms_comparison(save_dir='visualizations'):
    """对比三种算法的训练性能"""
    print("\n📊 加载三种算法的训练数据...")
    
    algos = ["MADDPG", "IDDPG", "MAPPO"]
    algo_colors = {
        "MADDPG": "#1f77b4",  # 蓝色
        "IDDPG": "#ff7f0e",   # 橙色
        "MAPPO": "#2ca02c"    # 绿色
    }
    algo_markers = {
        "MADDPG": "o",
        "IDDPG": "s",
        "MAPPO": "^"
    }
    
    # 加载所有算法数据
    data_dict = {}
    for algo in algos:
        df = load_training_data(algo)
        if df is not None:
            data_dict[algo] = df
    
    if len(data_dict) == 0:
        print("❌ 没有找到任何算法的训练数据！")
        return
    
    print(f"\n📈 开始绘制对比图表（共{len(data_dict)}个算法）...")
    
    os.makedirs(save_dir, exist_ok=True)
    num_agents = Config.NUM_AGENTS
    window_size = 50
    
    # 创建图表 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'多智能体算法对比 ({num_agents} Agents)', fontsize=18, fontweight='bold', y=0.995)
    
    # ==================== 1. 总奖励对比 ====================
    ax1 = axes[0, 0]
    for algo, df in data_dict.items():
        episodes = df['episode'].values
        total_reward = df['total_reward'].values
        
        if len(total_reward) >= window_size:
            ma_episodes = episodes[window_size-1:]
            ma_reward = moving_average(total_reward, window_size)
            ax1.plot(ma_episodes, ma_reward, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
        else:
            ax1.plot(episodes, total_reward, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax1.set_ylabel('总奖励', fontsize=12, fontweight='bold')
    ax1.set_title('总奖励对比', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=11)
    
    # ==================== 2. Agent成功率对比 ====================
    ax2 = axes[0, 1]
    for algo, df in data_dict.items():
        episodes = df['episode'].values
        success_rate = df['num_success'].values / num_agents * 100
        
        if len(success_rate) >= window_size:
            ma_episodes = episodes[window_size-1:]
            ma_success = moving_average(success_rate, window_size)
            ax2.plot(ma_episodes, ma_success, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
        else:
            ax2.plot(episodes, success_rate, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
    
    ax2.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Agent成功率 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Agent成功率对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11)
    ax2.set_ylim(0, 105)
    
    # ==================== 3. 全部成功率对比 ====================
    ax3 = axes[0, 2]
    for algo, df in data_dict.items():
        episodes = df['episode'].values
        all_success = df['all_success'].values * 100
        
        # 累计全部成功率
        cumsum_success = np.cumsum(all_success / 100)
        all_success_rate = cumsum_success / np.arange(1, len(all_success) + 1) * 100
        
        ax3.plot(episodes, all_success_rate, color=algo_colors[algo], 
                linewidth=2.5, label=algo, alpha=0.8)
    
    ax3.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax3.set_ylabel('全部成功率 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('全部Agent成功率对比', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=11)
    ax3.set_ylim(0, 105)
    
    # ==================== 4. 碰撞率对比 ====================
    ax4 = axes[1, 0]
    for algo, df in data_dict.items():
        episodes = df['episode'].values
        collision_rate = df['num_collision'].values / num_agents * 100
        
        if len(collision_rate) >= window_size:
            ma_episodes = episodes[window_size-1:]
            ma_collision = moving_average(collision_rate, window_size)
            ax4.plot(ma_episodes, ma_collision, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
        else:
            ax4.plot(episodes, collision_rate, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
    
    ax4.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax4.set_ylabel('碰撞率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Agent碰撞率对比', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='best', fontsize=11)
    ax4.set_ylim(0, 105)
    
    # ==================== 5. 步数对比 ====================
    ax5 = axes[1, 1]
    for algo, df in data_dict.items():
        episodes = df['episode'].values
        steps = df['steps'].values
        
        if len(steps) >= window_size:
            ma_episodes = episodes[window_size-1:]
            ma_steps = moving_average(steps, window_size)
            ax5.plot(ma_episodes, ma_steps, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
        else:
            ax5.plot(episodes, steps, color=algo_colors[algo], 
                    linewidth=2.5, label=algo, alpha=0.8)
    
    ax5.set_xlabel('训练轮次', fontsize=12, fontweight='bold')
    ax5.set_ylabel('步数', fontsize=12, fontweight='bold')
    ax5.set_title('每轮步数对比', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='best', fontsize=11)
    
    # ==================== 6. 性能统计对比 ====================
    ax6 = axes[1, 2]
    
    # 准备柱状图数据
    algo_names = []
    final_rewards = []
    final_success_rates = []
    final_all_success = []
    
    for algo, df in data_dict.items():
        algo_names.append(algo)
        
        # 计算最终100轮的统计
        total_reward = df['total_reward'].values
        success_rate = df['num_success'].values / num_agents * 100
        all_success = df['all_success'].values * 100
        cumsum_success = np.cumsum(all_success / 100)
        all_success_rate = cumsum_success / np.arange(1, len(all_success) + 1) * 100
        
        final_reward = total_reward[-100:].mean() if len(total_reward) >= 100 else total_reward.mean()
        final_success = success_rate[-100:].mean() if len(success_rate) >= 100 else success_rate.mean()
        final_all_succ = all_success_rate[-1] if len(all_success_rate) > 0 else 0
        
        final_rewards.append(final_reward)
        final_success_rates.append(final_success)
        final_all_success.append(final_all_succ)
    
    # 绘制对比柱状图
    x = np.arange(len(algo_names))
    width = 0.25
    
    bars1 = ax6.bar(x - width, final_success_rates, width, 
                    label='Agent成功率(%)', color='#2ca02c', alpha=0.8)
    bars2 = ax6.bar(x, final_all_success, width, 
                    label='全部成功率(%)', color='#1f77b4', alpha=0.8)
    bars3 = ax6.bar(x + width, [r/10 for r in final_rewards], width, 
                    label='总奖励(/10)', color='#ff7f0e', alpha=0.8)
    
    ax6.set_xlabel('算法', fontsize=12, fontweight='bold')
    ax6.set_ylabel('性能指标', fontsize=12, fontweight='bold')
    ax6.set_title('综合性能对比', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(algo_names, fontsize=11, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'algorithms_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {save_path}")
    
    plt.show()
    
    # 打印对比统计
    print("\n" + "="*80)
    print("📊 算法性能对比统计")
    print("="*80)
    print(f"{'算法':<10} {'训练轮次':<10} {'平均奖励':<12} {'Agent成功率':<15} {'全部成功率':<15}")
    print("-"*80)
    
    for i, algo in enumerate(algo_names):
        df = data_dict[algo]
        print(f"{algo:<10} {len(df):<10} {final_rewards[i]:>11.1f} {final_success_rates[i]:>13.1f}% {final_all_success[i]:>13.1f}%")
    
    print("="*80)
    
    # 找出最佳算法
    best_reward_idx = np.argmax(final_rewards)
    best_success_idx = np.argmax(final_success_rates)
    best_all_success_idx = np.argmax(final_all_success)
    
    print(f"\n🏆 最佳性能:")
    print(f"   最高平均奖励: {algo_names[best_reward_idx]} ({final_rewards[best_reward_idx]:.1f})")
    print(f"   最高Agent成功率: {algo_names[best_success_idx]} ({final_success_rates[best_success_idx]:.1f}%)")
    print(f"   最高全部成功率: {algo_names[best_all_success_idx]} ({final_all_success[best_all_success_idx]:.1f}%)")
    print("="*80)


def main(algo="MADDPG", compare=False):
    print("🎨 多智能体训练指标可视化")
    print("="*60)
    
    # 显示当前配置
    start_mode = "固定" if Config.AGENT_START_POSITIONS is not None else "随机"
    target_mode = "固定" if Config.AGENT_TARGET_POSITIONS is not None else "随机"
    obstacle_mode = "随机" if Config.USE_RANDOM_OBSTACLES else "固定"
    
    if compare:
        print(f"模式: 算法对比（MADDPG vs IDDPG vs MAPPO）")
    else:
        print(f"算法: {algo}")
    
    print(f"配置: 起点={start_mode}, 终点={target_mode}, 障碍物={obstacle_mode}")
    print("="*60)
    
    if compare:
        # 对比模式：绘制三个算法的对比图
        plot_algorithms_comparison()
    else:
        # 单算法模式：绘制详细图表
        plot_multi_agent_training_metrics(algo=algo)
    
    print("\n✅ 完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制训练指标')
    parser.add_argument('--algo', type=str, default='MADDPG',
                       choices=['MADDPG', 'MAPPO', 'IDDPG'],
                       help='算法类型（默认: MADDPG）')
    parser.add_argument('--compare', action='store_true',
                       help='对比模式：同时绘制三种算法的对比图')
    args = parser.parse_args()
    
    main(args.algo, args.compare)
