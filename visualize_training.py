#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多智能体训练数据可视化工具 - 独立运行脚本

使用方法：
  python visualize_training.py              # 生成所有图表
  python visualize_training.py --plot_type rewards  # 只生成奖励曲线
  python visualize_training.py --log_file custom_log.csv  # 使用自定义日志
  python visualize_training.py --output_dir my_plots  # 设置输出目录
"""

import os
import argparse
from training_visualizer import TrainingVisualizer
import matplotlib.pyplot as plt


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='多智能体训练数据可视化')
    parser.add_argument('--log_file', type=str, help='训练日志CSV文件路径')
    parser.add_argument('--output_dir', type=str, help='图表输出目录')
    parser.add_argument('--plot_type', type=str, 
                       choices=['all', 'rewards', 'success', 'collision', 'agents', 'steps', 'summary'],
                       default='all', help='要生成的图表类型')
    args = parser.parse_args()
    
    print(" 多智能体训练数据可视化工具")
    print("=" * 60)
    
    visualizer = TrainingVisualizer(statistics_dir=args.output_dir)
    headers, data_dict = visualizer.load_training_data(args.log_file)
    
    if data_dict is None:
        print(" 无法加载数据，退出程序")
        return
    
    print(f" 数据加载成功，共 {len(data_dict['episodes'])} 轮训练记录")
    
    output_dir = args.output_dir or visualizer.statistics_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if args.plot_type == 'all':
        visualizer.generate_all_plots(args.log_file)
        print("\n 所有图表生成完成!")
    else:
        plot_funcs = {
            'rewards': (visualizer.plot_rewards, "rewards.png"),
            'success': (visualizer.plot_success_rates, "success_rates.png"),
            'collision': (visualizer.plot_collision_rate, "collision_rate.png"),
            'agents': (visualizer.plot_agent_rewards, "agent_rewards.png"),
            'steps': (visualizer.plot_steps, "steps.png"),
            'summary': (visualizer.generate_summary_plots, "summary.png"),
        }
        
        func, filename = plot_funcs[args.plot_type]
        save_path = os.path.join(output_dir, filename)
        func(data_dict, save_path=save_path)
        print(f"\n {args.plot_type} 图表生成完成!")
        plt.show()


if __name__ == "__main__":
    main()
