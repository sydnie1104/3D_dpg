# training_visualizer.py - 多智能体训练可视化工具
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from config import Config

plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """多智能体训练数据可视化工具"""
    
    def __init__(self, log_dir=None, statistics_dir=None):
        self.log_dir = log_dir or Config.get_log_dir()
        self.statistics_dir = statistics_dir or "statistics"
        self.num_agents = Config.NUM_AGENTS
        os.makedirs(self.statistics_dir, exist_ok=True)
    
    def load_training_data(self, csv_log_path=None):
        """加载训练数据"""
        if csv_log_path is None:
            csv_log_path = os.path.join(self.log_dir, "training_metrics.csv")
        
        if not os.path.exists(csv_log_path):
            print(f"警告: 日志文件 {csv_log_path} 不存在")
            return None, None
        
        with open(csv_log_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        
        if len(data) <= 1:
            print(f"警告: 日志文件数据不足")
            return None, None
        
        headers = data[0]
        
        # 解析数据
        data_dict = {
            "episodes": [],
            "total_rewards": [],
            "avg_rewards": [],
            "num_success": [],
            "num_collision": [],
            "all_success": [],
            "steps": [],
            "noise_scale": [],
            "agent_rewards": [[] for _ in range(self.num_agents)]
        }
        
        for row in data[1:]:
            if len(row) < 8:
                continue
            
            data_dict["episodes"].append(int(row[0]))
            data_dict["total_rewards"].append(float(row[1]))
            data_dict["avg_rewards"].append(float(row[2]))
            
            # 各agent奖励
            idx = 3
            for i in range(self.num_agents):
                if idx < len(row):
                    data_dict["agent_rewards"][i].append(float(row[idx]))
                    idx += 1
            
            # 统计信息
            if idx < len(row):
                data_dict["num_success"].append(int(row[idx]))
                idx += 1
            if idx < len(row):
                data_dict["num_collision"].append(int(row[idx]))
                idx += 1
            if idx < len(row):
                data_dict["all_success"].append(int(row[idx]))
                idx += 1
            if idx < len(row):
                data_dict["steps"].append(int(row[idx]))
                idx += 1
            if idx < len(row):
                data_dict["noise_scale"].append(float(row[idx]))
        
        return headers, data_dict
    
    def plot_rewards(self, data_dict, smooth_window=50, save_path=None):
        """绘制奖励曲线"""
        plt.figure(figsize=(12, 6))
        
        episodes = data_dict["episodes"]
        rewards = data_dict["total_rewards"]
        
        plt.plot(episodes, rewards, alpha=0.3, label='总奖励')
        
        if smooth_window > 1 and len(rewards) >= smooth_window:
            smoothed = np.convolve(rewards, np.ones(smooth_window)/smooth_window, mode='valid')
            smoothed_ep = episodes[smooth_window-1:]
            plt.plot(smoothed_ep, smoothed, 'r-', linewidth=2, 
                    label=f'{smooth_window}轮移动平均')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('多智能体总奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('总奖励')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"奖励曲线已保存: {save_path}")
        
        return plt
    
    def plot_success_rates(self, data_dict, save_path=None):
        """绘制成功率曲线"""
        plt.figure(figsize=(12, 6))
        
        episodes = data_dict["episodes"]
        
        # Agent成功率
        agent_success_rate = np.array(data_dict["num_success"]) / self.num_agents * 100
        
        # 全部成功率
        all_success = np.array(data_dict["all_success"])
        cumsum = np.cumsum(all_success)
        all_success_rate = cumsum / np.arange(1, len(all_success) + 1) * 100
        
        plt.plot(episodes, agent_success_rate, alpha=0.5, color='blue', label='Agent成功率')
        plt.plot(episodes, all_success_rate, color='green', linewidth=2, label='累计全部成功率')
        
        plt.title('成功率曲线')
        plt.xlabel('Episode')
        plt.ylabel('成功率 (%)')
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"成功率曲线已保存: {save_path}")
        
        return plt
    
    def plot_collision_rate(self, data_dict, save_path=None):
        """绘制碰撞率曲线"""
        plt.figure(figsize=(12, 6))
        
        episodes = data_dict["episodes"]
        collision_rate = np.array(data_dict["num_collision"]) / self.num_agents * 100
        
        plt.plot(episodes, collision_rate, alpha=0.5, color='red', label='碰撞率')
        
        # 移动平均
        if len(collision_rate) >= 50:
            smoothed = np.convolve(collision_rate, np.ones(50)/50, mode='valid')
            plt.plot(episodes[49:], smoothed, color='darkred', linewidth=2, label='50轮移动平均')
        
        plt.title('碰撞率曲线')
        plt.xlabel('Episode')
        plt.ylabel('碰撞率 (%)')
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"碰撞率曲线已保存: {save_path}")
        
        return plt
    
    def plot_agent_rewards(self, data_dict, save_path=None):
        """绘制各Agent奖励对比"""
        plt.figure(figsize=(12, 6))
        
        episodes = data_dict["episodes"]
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        for i in range(self.num_agents):
            if i < len(data_dict["agent_rewards"]) and data_dict["agent_rewards"][i]:
                rewards = data_dict["agent_rewards"][i]
                if len(rewards) >= 50:
                    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
                    plt.plot(episodes[49:len(smoothed)+49], smoothed, 
                            color=colors[i % len(colors)], linewidth=1.5, 
                            label=f'Agent {i}', alpha=0.8)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('各Agent奖励对比 (50轮移动平均)')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Agent奖励对比已保存: {save_path}")
        
        return plt
    
    def plot_steps(self, data_dict, save_path=None):
        """绘制步数曲线"""
        plt.figure(figsize=(12, 6))
        
        episodes = data_dict["episodes"]
        steps = data_dict["steps"]
        
        plt.plot(episodes, steps, alpha=0.5, color='purple', label='每轮步数')
        
        if len(steps) >= 50:
            smoothed = np.convolve(steps, np.ones(50)/50, mode='valid')
            plt.plot(episodes[49:], smoothed, color='darkviolet', linewidth=2, label='50轮移动平均')
        
        plt.title('每轮步数曲线')
        plt.xlabel('Episode')
        plt.ylabel('步数')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"步数曲线已保存: {save_path}")
        
        return plt
    
    def generate_summary_plots(self, data_dict, save_path=None):
        """生成汇总图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'MADDPG训练汇总 ({self.num_agents} Agents)', fontsize=14, fontweight='bold')
        
        episodes = data_dict["episodes"]
        
        # 1. 总奖励
        ax = axes[0, 0]
        ax.plot(episodes, data_dict["total_rewards"], alpha=0.3)
        if len(data_dict["total_rewards"]) >= 50:
            smoothed = np.convolve(data_dict["total_rewards"], np.ones(50)/50, mode='valid')
            ax.plot(episodes[49:], smoothed, 'r-', linewidth=2)
        ax.set_title('总奖励')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        # 2. 成功率
        ax = axes[0, 1]
        success_rate = np.array(data_dict["num_success"]) / self.num_agents * 100
        ax.plot(episodes, success_rate, alpha=0.5)
        ax.set_title('Agent成功率 (%)')
        ax.set_xlabel('Episode')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # 3. 全部成功率
        ax = axes[0, 2]
        all_success = np.array(data_dict["all_success"])
        cumsum = np.cumsum(all_success)
        all_rate = cumsum / np.arange(1, len(all_success) + 1) * 100
        ax.plot(episodes, all_rate, 'g-', linewidth=2)
        ax.set_title('累计全部成功率 (%)')
        ax.set_xlabel('Episode')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # 4. 碰撞率
        ax = axes[1, 0]
        collision_rate = np.array(data_dict["num_collision"]) / self.num_agents * 100
        ax.plot(episodes, collision_rate, 'r', alpha=0.5)
        ax.set_title('碰撞率 (%)')
        ax.set_xlabel('Episode')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # 5. 步数
        ax = axes[1, 1]
        ax.plot(episodes, data_dict["steps"], 'purple', alpha=0.5)
        ax.set_title('每轮步数')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        # 6. 噪声衰减
        ax = axes[1, 2]
        if data_dict["noise_scale"]:
            ax.plot(episodes[:len(data_dict["noise_scale"])], data_dict["noise_scale"], 'orange')
        ax.set_title('噪声强度')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"汇总图表已保存: {save_path}")
        
        return plt
    
    def generate_all_plots(self, csv_log_path=None):
        """生成所有图表"""
        headers, data_dict = self.load_training_data(csv_log_path)
        if data_dict is None:
            return
        
        self.plot_rewards(data_dict, save_path=os.path.join(self.statistics_dir, "rewards.png"))
        self.plot_success_rates(data_dict, save_path=os.path.join(self.statistics_dir, "success_rates.png"))
        self.plot_collision_rate(data_dict, save_path=os.path.join(self.statistics_dir, "collision_rate.png"))
        self.plot_agent_rewards(data_dict, save_path=os.path.join(self.statistics_dir, "agent_rewards.png"))
        self.plot_steps(data_dict, save_path=os.path.join(self.statistics_dir, "steps.png"))
        self.generate_summary_plots(data_dict, save_path=os.path.join(self.statistics_dir, "training_summary.png"))
        
        plt.close('all')
        print(f"\n所有图表已保存到: {self.statistics_dir}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='多智能体训练数据可视化')
    parser.add_argument('--log_file', type=str, help='训练日志CSV文件路径')
    parser.add_argument('--output_dir', type=str, help='图表输出目录')
    parser.add_argument('--plot_type', type=str, 
                       choices=['all', 'rewards', 'success', 'collision', 'agents', 'steps', 'summary'],
                       default='all', help='图表类型')
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(statistics_dir=args.output_dir)
    headers, data_dict = visualizer.load_training_data(args.log_file)
    
    if data_dict is None:
        print("无法加载数据")
        return
    
    output_dir = args.output_dir or visualizer.statistics_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if args.plot_type == 'all':
        visualizer.generate_all_plots(args.log_file)
    elif args.plot_type == 'rewards':
        visualizer.plot_rewards(data_dict, save_path=os.path.join(output_dir, "rewards.png"))
    elif args.plot_type == 'success':
        visualizer.plot_success_rates(data_dict, save_path=os.path.join(output_dir, "success_rates.png"))
    elif args.plot_type == 'collision':
        visualizer.plot_collision_rate(data_dict, save_path=os.path.join(output_dir, "collision_rate.png"))
    elif args.plot_type == 'agents':
        visualizer.plot_agent_rewards(data_dict, save_path=os.path.join(output_dir, "agent_rewards.png"))
    elif args.plot_type == 'steps':
        visualizer.plot_steps(data_dict, save_path=os.path.join(output_dir, "steps.png"))
    elif args.plot_type == 'summary':
        visualizer.generate_summary_plots(data_dict, save_path=os.path.join(output_dir, "summary.png"))
    
    if args.plot_type != 'all':
        plt.show()


if __name__ == "__main__":
    main()
