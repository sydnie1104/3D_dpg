# orca_reward.py
import numpy as np
import math
from config import Config

EPS = 1e-8

def to_array(pos):
    if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
        return np.array([pos.x, pos.y, pos.z], dtype=float)
    elif isinstance(pos, (list, tuple, np.ndarray)):
        arr = np.array(pos, dtype=float).flatten()
        if arr.size >= 3:
            return arr[:3]
        res = np.zeros(3, dtype=float)
        res[:arr.size] = arr
        return res
    else:
        return np.zeros(3, dtype=float)

class ORCARewardCalculator:
    def __init__(self):
        self.robot_radius = 1.0
        self.time_horizon = 2.0
        self.max_speed = 3.0
        self.collision_penalty = -30
        self.avoidance_reward = 0.5  # 进一步降低，避免过度奖励
        self.goal_reward_weight = 2.5  # 大幅降低！从4.0降到1.0，避免过度奖励朝向目标
        self.safety_reward_weight = 0.3  # 保持安全奖励
        self.speed_reward_weight = 0.5
        self.desired_safety_distance = 1.5
        self.min_safety_distance = 0.8  # 调低最小安全距离

    def _calculate_time_to_collision(self, rel_pos, rel_vel, total_radius):
        v = np.linalg.norm(rel_vel)
        if v < EPS:
            return float('inf')
        projection = np.dot(rel_pos, rel_vel) / (v + EPS)
        if projection < 0:
            return float('inf')
        dist_sq = np.dot(rel_pos, rel_pos)
        perpendicular_dist_sq = dist_sq - projection * projection
        if perpendicular_dist_sq > total_radius * total_radius:
            return float('inf')
        parallel_dist = math.sqrt(max(0, total_radius * total_radius - perpendicular_dist_sq))
        ttc = (projection - parallel_dist) / (v + EPS)
        return max(0.0, ttc)

    def compute_orca_constraint(self, robot_pos, robot_vel, obstacle_pos, obstacle_vel, obstacle_radius):
        robot_pos = to_array(robot_pos)
        obstacle_pos = to_array(obstacle_pos)
        robot_vel = to_array(robot_vel)
        obstacle_vel = to_array(obstacle_vel)

        rel_pos = obstacle_pos - robot_pos
        rel_vel = obstacle_vel - robot_vel

        dist = np.linalg.norm(rel_pos) + EPS
        total_radius = self.robot_radius + obstacle_radius

        if dist > total_radius * 2:
            return None, 0.0

        ttc = self._calculate_time_to_collision(rel_pos, rel_vel, total_radius)
        collision_risk = 0.0
        if ttc <= self.time_horizon:
            collision_risk = max(0.0, 1.0 - ttc / self.time_horizon)

        n = rel_pos / dist
        dot_product = np.dot(rel_vel, n)
        w = total_radius / self.time_horizon
        c = np.dot(rel_pos, rel_pos) - w * w
        if c < 0:
            c = 0
        if dot_product < -math.sqrt(c + EPS):
            delta_v = (-dot_product + math.sqrt(c + EPS)) * n
            constraint = (n, delta_v)
            return constraint, collision_risk
        return None, collision_risk

    def calculate_safety_reward(self, robot_pos, dynamic_obstacles, robot_vel=None):
        if not dynamic_obstacles:
            return 0.0
        if robot_vel is None:
            robot_vel = np.zeros(3, dtype=float)
        total_collision_risk = 0.0
        num_constraints = 0
        for obs in dynamic_obstacles:
            obs_pos = getattr(obs, 'pos', obs)
            obs_vel = getattr(obs, 'velocity', np.zeros(3, dtype=float))
            obs_radius = getattr(obs, 'radius', 0.0)
            constraint, collision_risk = self.compute_orca_constraint(robot_pos, robot_vel, obs_pos, obs_vel, obs_radius)
            total_collision_risk += collision_risk
            if constraint is not None:
                num_constraints += 1
        avg_collision_risk = total_collision_risk / len(dynamic_obstacles)
        safety_reward = -self.safety_reward_weight * avg_collision_risk * self.avoidance_reward
        if num_constraints > 0:
            safety_reward += self.avoidance_reward * (num_constraints / len(dynamic_obstacles))
        return safety_reward

    def calculate_goal_reward(self, robot_pos, target_pos, robot_vel=None):
        robot_pos = to_array(robot_pos)
        target_pos = to_array(target_pos)
        goal_vec = target_pos - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist < EPS:
            return 0.0
        goal_dir = goal_vec / goal_dist
        if robot_vel is None:
            robot_vel = np.zeros(3, dtype=float)
        robot_vel = to_array(robot_vel)
        robot_speed = np.linalg.norm(robot_vel)
        if robot_speed < EPS:
            return -0.5 * self.goal_reward_weight * (goal_dist / Config.GRID_SCALE)
        robot_dir = robot_vel / (robot_speed + EPS)
        dot_product = np.clip(np.dot(robot_dir, goal_dir), -1.0, 1.0)
        direction_reward = self.goal_reward_weight * dot_product
        distance_reward = -self.goal_reward_weight * (goal_dist / Config.GRID_SCALE)
        return direction_reward + distance_reward

    def calculate_speed_reward(self, robot_vel):
        robot_vel = to_array(robot_vel)
        speed = np.linalg.norm(robot_vel)
        if speed > self.max_speed:
            return -self.speed_reward_weight * ((speed - self.max_speed) / self.max_speed)
        elif speed < 0.1:
            return -self.speed_reward_weight * 0.3
        normalized_speed = speed / self.max_speed
        return self.speed_reward_weight * (1 - (normalized_speed - 0.5) ** 2)

    def calculate_orca_reward(self, robot_pos, robot_vel, target_pos, dynamic_obstacles):
        safety_reward = self.calculate_safety_reward(robot_pos, dynamic_obstacles, robot_vel)
        goal_reward = self.calculate_goal_reward(robot_pos, target_pos, robot_vel)
        speed_reward = self.calculate_speed_reward(robot_vel)
        total_reward = safety_reward + goal_reward + speed_reward
        return safety_reward + goal_reward

    def calculate_collision_penalty(self, has_collided, collision_type='dynamic'):
        if not has_collided:
            return 0.0
        if collision_type == 'dynamic':
            return self.collision_penalty
        elif collision_type == 'fixed':
            return Config.COLLISION_PENALTY
        elif collision_type == 'boundary':
            return Config.CROSS_BORDER_PENALTY
        else:
            return self.collision_penalty
