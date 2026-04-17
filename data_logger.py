# data_logger.py - 多智能体数据记录器
import os
import csv
from config import Config


class DataLogger:
    """多智能体训练数据记录器"""
    
    def __init__(self, append=False, log_dir=None):
        self.log_dir = log_dir if log_dir else Config.get_log_dir()
        self.num_agents = Config.NUM_AGENTS
        os.makedirs(self.log_dir, exist_ok=True)
        
        mode = 'a' if append else 'w'
        self.files = {}
        
        # 全局指标
        global_metrics = {
            "episode_reward": "episode_reward.csv",
            "success_rate": "success_rate.csv",
            "collision_rate": "collision_rate.csv",
            "all_success_rate": "all_success_rate.csv",
            "steps_per_episode": "steps_per_episode.csv",
            "planning_time": "planning_time.csv",
        }
        
        for k, fname in global_metrics.items():
            path = os.path.join(self.log_dir, fname)
            f = open(path, mode, newline='')
            self.files[k] = f
            if not append:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Value"])
        
        # 每个agent的指标
        for i in range(self.num_agents):
            agent_metrics = {
                f"agent_{i}_reward": f"agent_{i}_reward.csv",
                f"agent_{i}_success": f"agent_{i}_success.csv",
                f"agent_{i}_collision": f"agent_{i}_collision.csv",
                f"agent_{i}_final_distance": f"agent_{i}_final_distance.csv",
            }
            for k, fname in agent_metrics.items():
                path = os.path.join(self.log_dir, fname)
                f = open(path, mode, newline='')
                self.files[k] = f
                if not append:
                    writer = csv.writer(f)
                    writer.writerow(["Episode", "Value"])
    
    def get_writers(self):
        return {k: csv.writer(f) for k, f in self.files.items()}
    
    def log(self, episode, data_type, value):
        """记录单个指标"""
        writers = self.get_writers()
        if data_type in writers:
            writers[data_type].writerow([episode, value])
            self.files[data_type].flush()
    
    def log_episode(self, episode, rewards, successes, collisions, final_distances, 
                    total_reward, num_success, num_collision, all_success, steps, planning_time=0):
        """记录一个episode的所有指标"""
        writers = self.get_writers()
        
        # 全局指标
        writers["episode_reward"].writerow([episode, total_reward])
        writers["success_rate"].writerow([episode, num_success / self.num_agents])
        writers["collision_rate"].writerow([episode, num_collision / self.num_agents])
        writers["all_success_rate"].writerow([episode, 1 if all_success else 0])
        writers["steps_per_episode"].writerow([episode, steps])
        writers["planning_time"].writerow([episode, planning_time])
        
        # 每个agent的指标
        for i in range(self.num_agents):
            writers[f"agent_{i}_reward"].writerow([episode, rewards[i]])
            writers[f"agent_{i}_success"].writerow([episode, 1 if successes[i] else 0])
            writers[f"agent_{i}_collision"].writerow([episode, 1 if collisions[i] else 0])
            if i < len(final_distances):
                writers[f"agent_{i}_final_distance"].writerow([episode, final_distances[i]])
        
        # 刷新所有文件
        for f in self.files.values():
            f.flush()
    
    def close(self):
        for f in self.files.values():
            f.close()


if __name__ == '__main__':
    # 测试
    logger = DataLogger(append=False)
    
    # 模拟记录
    rewards = [100.0, 80.0, 90.0]
    successes = [True, False, True]
    collisions = [False, True, False]
    final_distances = [0.5, 5.0, 0.8]
    
    logger.log_episode(
        episode=1,
        rewards=rewards,
        successes=successes,
        collisions=collisions,
        final_distances=final_distances,
        total_reward=sum(rewards),
        num_success=sum(successes),
        num_collision=sum(collisions),
        all_success=all(successes),
        steps=100,
        planning_time=0.01
    )
    
    logger.close()
    print("数据记录器测试完成!")
