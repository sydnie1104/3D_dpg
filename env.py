# env.py - 多智能体3D无人机避障环境（MADDPG）
import numpy as np
import math
from collections import deque
from config import Config


class Point:
    """3D点/位置类"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def to_array(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    def copy(self):
        return Point(self.x, self.y, self.z)

    def __add__(self, other):
        arr = self.to_array() + np.array(other, dtype=float)
        return Point(*arr)

    def __sub__(self, other):
        arr = self.to_array() - np.array(other, dtype=float)
        return Point(*arr)

    def __repr__(self):
        return f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]"


class DynamicObstacle:
    """动态障碍物类"""
    def __init__(self, x, y, z, radius, vx, vy, vz, grid_scale):
        self.pos = Point(x, y, z)
        self.radius = radius
        self.velocity = np.array([vx, vy, vz], dtype=float)
        self.grid_scale = grid_scale

    def update(self):
        arr = self.pos.to_array() + self.velocity
        for i in range(3):
            if arr[i] - self.radius <= 0 or arr[i] + self.radius >= self.grid_scale:
                self.velocity[i] = -self.velocity[i]
                arr[i] = np.clip(arr[i], self.radius, self.grid_scale - self.radius)
        self.pos = Point(*arr)

    def get_info(self):
        return [*self.pos.to_array(), self.radius, *self.velocity]


class MultiAgentDroneEnv:
    """多智能体无人机环境"""
    
    def __init__(self):
        # 基本参数
        self.num_agents = Config.NUM_AGENTS
        self.grid_scale = Config.GRID_SCALE
        self.max_steps = Config.MAX_STEPS
        self.d_safe = Config.D_SAFE
        self.action_scale = Config.ACTION_SCALE
        
        # 奖励参数
        self.collision_penalty = Config.COLLISION_PENALTY
        self.success_reward = Config.SUCCESS_REWARD
        self.all_success_bonus = Config.ALL_SUCCESS_BONUS
        self.timeout_penalty = Config.TIMEOUT_PENALTY
        self.agent_collision_penalty = Config.AGENT_COLLISION_PENALTY
        self.cross_border_penalty = Config.CROSS_BORDER_PENALTY
        
        # Agent安全距离
        self.agent_collision_dist = Config.AGENT_COLLISION_DIST
        self.agent_safe_dist = Config.AGENT_SAFE_DIST
        
        # 静态障碍物
        self.use_random_obstacles = Config.USE_RANDOM_OBSTACLES
        if self.use_random_obstacles:
            self.obstacles = self._generate_random_static_obstacles()
        else:
            self.obstacles = Config.FIXED_OBSTACLES
        
        # 动态障碍物
        self.dynamic_obstacles = self._generate_dynamic_obstacles()
        
        # 多agent状态
        self.drone_positions = []  # List[Point]
        self.target_positions = []  # List[Point]
        self.velocities = []  # List[np.array]
        self.path_histories = []  # List[List[Point]]
        self.action_histories = []  # List[deque]
        self.agent_done = []  # 每个agent是否完成
        self.agent_success = []  # 每个agent是否成功到达
        self.agent_collided = []  # 每个agent是否碰撞
        
        # 全局状态
        self.step_count = 0
        self.episode_done = False
        
        # 初始化agent状态
        self._init_agents()
        
        # 观测维度
        self.obs_dim = Config.get_single_obs_dim()
        self.action_dim = Config.get_action_dim()
        
    def _init_agents(self):
        """初始化所有agent的位置和目标"""
        self.drone_positions = []
        self.target_positions = []
        self.velocities = []
        self.path_histories = []
        self.action_histories = []
        self.agent_done = [False] * self.num_agents
        self.agent_success = [False] * self.num_agents
        self.agent_collided = [False] * self.num_agents
        
        # 生成起点
        if Config.AGENT_START_POSITIONS is not None:
            for pos in Config.AGENT_START_POSITIONS[:self.num_agents]:
                self.drone_positions.append(Point(*pos))
        else:
            self.drone_positions = self._generate_random_positions(self.num_agents, margin=3.0)
        
        # 生成终点
        if Config.AGENT_TARGET_POSITIONS is not None:
            for pos in Config.AGENT_TARGET_POSITIONS[:self.num_agents]:
                self.target_positions.append(Point(*pos))
        else:
            self.target_positions = self._generate_random_positions(
                self.num_agents, margin=3.0, 
                avoid_positions=[p.to_array() for p in self.drone_positions]
            )
        
        # 初始化速度、路径历史、动作历史
        for i in range(self.num_agents):
            self.velocities.append(np.zeros(3))
            self.path_histories.append([self.drone_positions[i].copy()])
            self.action_histories.append(deque(maxlen=Config.ACTION_HISTORY_LEN))
    
    def _generate_random_positions(self, n, margin=3.0, avoid_positions=None, min_dist=4.0):
        """生成n个随机位置，确保相互之间有足够距离"""
        positions = []
        avoid = avoid_positions if avoid_positions else []
        
        for _ in range(n):
            for attempt in range(100):
                x = np.random.uniform(margin, self.grid_scale - margin)
                y = np.random.uniform(margin, self.grid_scale - margin)
                z = np.random.uniform(margin, self.grid_scale - margin)
                pos = np.array([x, y, z])
                
                # 检查与已有位置的距离
                valid = True
                for existing in positions + avoid:
                    # 处理Point对象和数组
                    if hasattr(existing, 'to_array'):
                        existing_arr = existing.to_array()
                    elif isinstance(existing, np.ndarray):
                        existing_arr = existing
                    else:
                        existing_arr = np.array(existing)
                    if np.linalg.norm(pos - existing_arr) < min_dist:
                        valid = False
                        break
                
                # 检查与静态障碍物的距离
                if valid:
                    for obs in self.obstacles:
                        if self._point_to_obs_distance(x, y, z, obs) < 2.0:
                            valid = False
                            break
                
                if valid:
                    positions.append(Point(x, y, z))
                    break
            else:
                # 如果100次尝试都失败，强制放置
                x = np.random.uniform(margin, self.grid_scale - margin)
                y = np.random.uniform(margin, self.grid_scale - margin)
                z = np.random.uniform(margin, self.grid_scale - margin)
                positions.append(Point(x, y, z))
        
        return positions
    
    def _generate_random_static_obstacles(self):
        """生成随机静态障碍物"""
        obstacles = []
        num_obstacles = 25
        
        for _ in range(num_obstacles):
            for attempt in range(100):
                x = np.random.uniform(3.0, self.grid_scale - 3.0)
                y = np.random.uniform(3.0, self.grid_scale - 3.0)
                bd_w = np.random.uniform(0.3, 0.8)
                bd_l = np.random.uniform(0.3, 0.8)
                h = np.random.uniform(8.0, 18.0)
                obstacles.append((x, y, bd_w, bd_l, h))
                break
        
        return obstacles
    
    def _generate_dynamic_obstacles(self):
        """生成动态障碍物"""
        dynamic_obstacles = []
        
        if hasattr(Config, 'CUSTOM_DYNAMIC_OBSTACLES') and Config.CUSTOM_DYNAMIC_OBSTACLES:
            for cfg in Config.CUSTOM_DYNAMIC_OBSTACLES:
                x, y, z, radius, vx, vy, vz = cfg
                dynamic_obstacles.append(DynamicObstacle(x, y, z, radius, vx, vy, vz, self.grid_scale))
            return dynamic_obstacles
        
        margin = max(2, int(self.grid_scale * 0.1))
        
        for _ in range(Config.NUM_DYNAMIC_OBSTACLES):
            for attempt in range(100):
                x = np.random.uniform(margin, self.grid_scale - margin)
                y = np.random.uniform(margin, self.grid_scale - margin)
                z = np.random.uniform(margin, self.grid_scale - margin)
                
                radius = np.random.uniform(Config.DYNAMIC_OBSTACLE_MIN_RADIUS, Config.DYNAMIC_OBSTACLE_MAX_RADIUS)
                speed = np.random.uniform(Config.DYNAMIC_OBSTACLE_MIN_SPEED, Config.DYNAMIC_OBSTACLE_MAX_SPEED)
                
                theta = np.random.uniform(0, 2 * math.pi)
                phi = np.random.uniform(0, math.pi)
                vx = speed * math.sin(phi) * math.cos(theta)
                vy = speed * math.sin(phi) * math.sin(theta)
                vz = speed * math.cos(phi)
                
                dynamic_obstacles.append(DynamicObstacle(x, y, z, radius, vx, vy, vz, self.grid_scale))
                break
        
        return dynamic_obstacles
    
    def _point_to_obs_distance(self, px, py, pz, obs):
        """计算点到静态障碍物的距离"""
        ox, oy, bd_w, bd_l, h = obs
        
        x_min, x_max = ox - bd_w / 2.0, ox + bd_w / 2.0
        y_min, y_max = oy - bd_l / 2.0, oy + bd_l / 2.0
        z_min, z_max = 0.0, h
        
        if (x_min <= px <= x_max) and (y_min <= py <= y_max) and (z_min <= pz <= z_max):
            return 0.0
        
        dx = max(x_min - px, px - x_max, 0.0)
        dy = max(y_min - py, py - y_max, 0.0)
        dz = max(z_min - pz, pz - z_max, 0.0)
        
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_distance(self, agent_idx):
        """获取指定agent到其目标的距离"""
        return np.linalg.norm(
            self.target_positions[agent_idx].to_array() - 
            self.drone_positions[agent_idx].to_array()
        )
    
    def _check_static_collision(self, pos_array):
        """检查与静态障碍物的碰撞"""
        x, y, z = pos_array
        for obs in self.obstacles:
            if self._point_to_obs_distance(x, y, z, obs) <= 0.0:
                return True
        return False
    
    def _check_dynamic_collision(self, pos_array):
        """检查与动态障碍物的碰撞"""
        for dyn_obs in self.dynamic_obstacles:
            dist = np.linalg.norm(pos_array - dyn_obs.pos.to_array())
            if dist <= dyn_obs.radius:
                return True
        return False
    
    def _check_boundary(self, pos_array):
        """检查是否越界"""
        return np.any(pos_array < 0) or np.any(pos_array > self.grid_scale)
    
    def _check_agent_collision(self, agent_idx, pos_array):
        """检查与其他agent的碰撞"""
        for j in range(self.num_agents):
            if j != agent_idx and not self.agent_done[j]:
                other_pos = self.drone_positions[j].to_array()
                dist = np.linalg.norm(pos_array - other_pos)
                if dist < self.agent_collision_dist:
                    return True
        return False
    
    def _ray_casting(self, agent_idx):
        """Ray Casting: 从agent位置向多个方向发射射线"""
        directions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]
        
        ray_results = []
        drone_pos = self.drone_positions[agent_idx].to_array()
        max_ray_length = self.grid_scale * 1.5
        
        for direction in directions:
            dir_vec = np.array(direction, dtype=float)
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            min_dist = max_ray_length
            obstacle_type = 0.0  # 0=无障碍, 1=静态, 2=动态, 3=边界, 4=其他agent
            
            # 检测边界
            for i in range(3):
                if dir_vec[i] > 0:
                    t = (self.grid_scale - drone_pos[i]) / (dir_vec[i] + 1e-8)
                elif dir_vec[i] < 0:
                    t = -drone_pos[i] / (dir_vec[i] - 1e-8)
                else:
                    t = max_ray_length
                if 0 < t < min_dist:
                    min_dist = t
                    obstacle_type = 3.0
            
            # 检测静态障碍物
            for obs in self.obstacles:
                t = self._ray_box_intersection(drone_pos, dir_vec, obs)
                if t is not None and 0 < t < min_dist:
                    min_dist = t
                    obstacle_type = 1.0
            
            # 检测动态障碍物
            for dyn_obs in self.dynamic_obstacles:
                t = self._ray_sphere_intersection(drone_pos, dir_vec, dyn_obs.pos.to_array(), dyn_obs.radius)
                if t is not None and 0 < t < min_dist:
                    min_dist = t
                    obstacle_type = 2.0
            
            # 检测其他agent（视为球体）
            for j in range(self.num_agents):
                if j != agent_idx and not self.agent_done[j]:
                    other_pos = self.drone_positions[j].to_array()
                    t = self._ray_sphere_intersection(drone_pos, dir_vec, other_pos, self.agent_collision_dist)
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
                        obstacle_type = 4.0
            
            normalized_dist = min(min_dist / max_ray_length, 1.0)
            ray_results.extend([normalized_dist, obstacle_type / 4.0])  # 类型归一化到[0,1]
        
        return np.array(ray_results)
    
    def _ray_box_intersection(self, origin, direction, box):
        """射线与长方体相交检测"""
        ox, oy, bd_w, bd_l, h = box
        box_min = np.array([ox - bd_w/2, oy - bd_l/2, 0.0])
        box_max = np.array([ox + bd_w/2, oy + bd_l/2, h])
        
        t_min = -np.inf
        t_max = np.inf
        
        for i in range(3):
            if abs(direction[i]) < 1e-8:
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return None
            else:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        
        if t_min > t_max or t_max < 0:
            return None
        return t_min if t_min > 0 else t_max
    
    def _ray_sphere_intersection(self, origin, direction, center, radius):
        """射线与球体相交检测"""
        oc = origin - center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return t if t > 0 else None
    
    def _compute_local_goal_vector(self, agent_idx):
        """计算局部终点向量（5维）"""
        drone_pos = self.drone_positions[agent_idx].to_array()
        target_pos = self.target_positions[agent_idx].to_array()
        goal_vec = target_pos - drone_pos
        goal_dist = np.linalg.norm(goal_vec)
        
        if goal_dist < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        goal_dir = goal_vec / goal_dist
        
        # 1. 检测目标是否可见
        is_visible = 1.0
        for obs in self.obstacles:
            t = self._ray_box_intersection(drone_pos, goal_dir, obs)
            if t is not None and 0 < t < goal_dist:
                is_visible = 0.0
                break
        
        if is_visible > 0.5:
            for dyn_obs in self.dynamic_obstacles:
                t = self._ray_sphere_intersection(drone_pos, goal_dir, dyn_obs.pos.to_array(), dyn_obs.radius)
                if t is not None and 0 < t < goal_dist:
                    is_visible = 0.0
                    break
        
        # 检测其他agent是否阻挡
        if is_visible > 0.5:
            for j in range(self.num_agents):
                if j != agent_idx and not self.agent_done[j]:
                    other_pos = self.drone_positions[j].to_array()
                    t = self._ray_sphere_intersection(drone_pos, goal_dir, other_pos, self.agent_collision_dist)
                    if t is not None and 0 < t < goal_dist:
                        is_visible = 0.0
                        break
        
        # 2. 计算最近射线方向与目标方向的夹角
        directions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]
        
        min_angle = np.pi
        closest_ray_dist = 1.0
        max_ray_length = self.grid_scale * 1.5
        
        for direction in directions:
            dir_vec = np.array(direction, dtype=float)
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            angle = np.arccos(np.clip(np.dot(dir_vec, goal_dir), -1.0, 1.0))
            
            if angle < min_angle:
                min_angle = angle
                min_dist = max_ray_length
                
                for obs in self.obstacles:
                    t = self._ray_box_intersection(drone_pos, dir_vec, obs)
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
                
                for dyn_obs in self.dynamic_obstacles:
                    t = self._ray_sphere_intersection(drone_pos, dir_vec, dyn_obs.pos.to_array(), dyn_obs.radius)
                    if t is not None and 0 < t < min_dist:
                        min_dist = t
                
                closest_ray_dist = min(min_dist / max_ray_length, 1.0)
        
        normalized_angle = min_angle / np.pi
        
        # 3. 计算俯仰角和方位角
        horizontal_dist = np.sqrt(goal_vec[0]**2 + goal_vec[1]**2)
        if horizontal_dist > 1e-6:
            pitch = np.arctan2(goal_vec[2], horizontal_dist) / (np.pi / 2)
        else:
            pitch = np.sign(goal_vec[2])
        
        if horizontal_dist > 1e-6:
            azimuth = np.arctan2(goal_vec[1], goal_vec[0]) / np.pi
        else:
            azimuth = 0.0
        
        return np.array([is_visible, normalized_angle, closest_ray_dist, pitch, azimuth])
    
    def _get_other_agents_info(self, agent_idx):
        """获取其他agent的信息"""
        other_info = []
        my_pos = self.drone_positions[agent_idx].to_array()
        my_vel = self.velocities[agent_idx]
        
        for j in range(self.num_agents):
            if j != agent_idx:
                other_pos = self.drone_positions[j].to_array()
                other_vel = self.velocities[j]
                
                # 相对位置（归一化）
                rel_pos = (other_pos - my_pos) / self.grid_scale
                # 相对速度（归一化）
                rel_vel = (other_vel - my_vel) / (Config.ACTION_SCALE * 2)
                # 距离（归一化）
                dist = np.array([np.linalg.norm(other_pos - my_pos) / self.grid_scale])
                
                other_info.extend([*rel_pos, *rel_vel, *dist])
        
        return np.array(other_info)
    
    def get_obs(self, agent_idx):
        """获取单个agent的观测"""
        drone_pos = self.drone_positions[agent_idx].to_array()
        target_pos = self.target_positions[agent_idx].to_array()
        
        # 1. 相对目标位置（归一化）
        rel_goal = (target_pos - drone_pos) / self.grid_scale
        
        # 2. 目标距离（归一化）
        goal_dist = np.array([self.get_distance(agent_idx) / self.grid_scale])
        
        # 3. 自身速度（归一化）
        vel = self.velocities[agent_idx] / (Config.ACTION_SCALE * 2)
        
        # 4. Ray Casting
        ray_info = self._ray_casting(agent_idx)
        
        # 5. 局部目标向量
        local_goal_info = self._compute_local_goal_vector(agent_idx)
        
        # 6. 动态障碍物信息
        dyn_info = []
        for dyn_obs in self.dynamic_obstacles:
            rel_pos = (dyn_obs.pos.to_array() - drone_pos) / self.grid_scale
            rel_vel = dyn_obs.velocity / Config.DYNAMIC_OBSTACLE_MAX_SPEED
            dyn_info.extend([*rel_pos, dyn_obs.radius / self.grid_scale, *rel_vel])
        
        while len(dyn_info) < Config.NUM_DYNAMIC_OBSTACLES * Config.DYN_OBS_INFO_DIM:
            dyn_info.extend([0.0] * Config.DYN_OBS_INFO_DIM)
        
        # 7. 动作历史
        action_hist = np.array(list(self.action_histories[agent_idx]), dtype=float)
        if len(action_hist) < Config.ACTION_HISTORY_LEN:
            padding = np.zeros(Config.ACTION_HISTORY_LEN - len(action_hist))
            action_hist = np.concatenate([padding, action_hist])
        
        # 8. 其他agent信息
        other_agents_info = self._get_other_agents_info(agent_idx)
        
        # 拼接所有观测
        obs = np.concatenate([
            rel_goal,           # 3
            goal_dist,          # 1
            vel,                # 3
            ray_info,           # 52
            local_goal_info,    # 5
            np.array(dyn_info), # 35
            action_hist,        # 5
            other_agents_info   # (num_agents-1) * 7
        ])
        
        return obs
    
    def get_all_obs(self):
        """获取所有agent的观测"""
        return [self.get_obs(i) for i in range(self.num_agents)]
    
    def get_global_state(self):
        """获取全局状态（用于Critic）"""
        all_obs = self.get_all_obs()
        return np.concatenate(all_obs)
    
    def compute_reward(self, agent_idx, action, new_pos, old_pos):
        """计算单个agent的奖励"""
        reward = 0.0
        info = {'collision': False, 'success': False, 'agent_collision': False, 'boundary': False}
        
        new_pos_arr = new_pos.to_array()
        old_pos_arr = old_pos.to_array()
        target_pos_arr = self.target_positions[agent_idx].to_array()
        
        # 1. 检查越界
        if self._check_boundary(new_pos_arr):
            reward += self.cross_border_penalty
            info['boundary'] = True
            self.agent_collided[agent_idx] = True
            self.agent_done[agent_idx] = True
            return reward, info
        
        # 2. 检查与静态障碍物碰撞
        if self._check_static_collision(new_pos_arr):
            reward += self.collision_penalty
            info['collision'] = True
            self.agent_collided[agent_idx] = True
            self.agent_done[agent_idx] = True
            return reward, info
        
        # 3. 检查与动态障碍物碰撞
        if self._check_dynamic_collision(new_pos_arr):
            reward += self.collision_penalty
            info['collision'] = True
            self.agent_collided[agent_idx] = True
            self.agent_done[agent_idx] = True
            return reward, info
        
        # 4. 检查与其他agent碰撞
        if self._check_agent_collision(agent_idx, new_pos_arr):
            reward += self.agent_collision_penalty
            info['agent_collision'] = True
            self.agent_collided[agent_idx] = True
            self.agent_done[agent_idx] = True
            return reward, info
        
        # 5. 检查是否到达目标
        dist_to_goal = np.linalg.norm(new_pos_arr - target_pos_arr)
        if dist_to_goal <= self.d_safe:
            reward += self.success_reward
            info['success'] = True
            self.agent_success[agent_idx] = True
            self.agent_done[agent_idx] = True
            return reward, info
        
        # 6. 进步奖励
        old_dist = np.linalg.norm(old_pos_arr - target_pos_arr)
        new_dist = dist_to_goal
        progress = old_dist - new_dist
        
        if progress > 0:
            reward += min(10.0, progress * 20.0)
        else:
            reward += max(-2.0, progress * 5.0)
        
        # 7. 距离惩罚（鼓励快速到达）
        reward -= new_dist / self.grid_scale * 0.5
        
        # 8. 与其他agent保持安全距离的奖励
        for j in range(self.num_agents):
            if j != agent_idx and not self.agent_done[j]:
                other_pos = self.drone_positions[j].to_array()
                dist_to_agent = np.linalg.norm(new_pos_arr - other_pos)
                if dist_to_agent < self.agent_safe_dist:
                    # 距离越近惩罚越大
                    penalty = -5.0 * (1.0 - dist_to_agent / self.agent_safe_dist)
                    reward += penalty
        
        # 9. 与障碍物的安全距离奖励
        min_obs_dist = float('inf')
        for obs in self.obstacles:
            d = self._point_to_obs_distance(*new_pos_arr, obs)
            min_obs_dist = min(min_obs_dist, d)
        
        if min_obs_dist < 1.0:
            reward -= 3.0 * (1.0 - min_obs_dist)
        
        # 10. 小的存活奖励
        reward += 0.1
        
        return reward, info
    
    def step(self, actions):
        """
        执行一步环境交互
        
        Args:
            actions: List[np.array], 每个agent的动作 [vx, vy, vz]
        
        Returns:
            obs_list: List[np.array], 每个agent的新观测
            rewards: List[float], 每个agent的奖励
            dones: List[bool], 每个agent是否结束
            info: dict, 附加信息
        """
        rewards = []
        infos = []
        
        # 保存旧位置
        old_positions = [p.copy() for p in self.drone_positions]
        
        # 更新动态障碍物
        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.update()
        
        # 处理每个agent的动作
        for i in range(self.num_agents):
            if self.agent_done[i]:
                # 已完成的agent保持静止，给0奖励
                rewards.append(0.0)
                infos.append({'collision': False, 'success': self.agent_success[i], 
                             'agent_collision': False, 'boundary': False})
                continue
            
            # 应用动作
            action = np.clip(actions[i], -1.0, 1.0)
            delta = action * self.action_scale
            new_pos_arr = self.drone_positions[i].to_array() + delta
            new_pos_arr = np.clip(new_pos_arr, 0.0, self.grid_scale)
            new_pos = Point(*new_pos_arr)
            
            # 更新速度
            self.velocities[i] = new_pos_arr - old_positions[i].to_array()
            
            # 计算奖励
            reward, info = self.compute_reward(i, action, new_pos, old_positions[i])
            rewards.append(reward)
            infos.append(info)
            
            # 更新位置
            self.drone_positions[i] = new_pos
            self.path_histories[i].append(new_pos.copy())
            
            # 记录动作幅度
            action_magnitude = np.linalg.norm(delta) / self.action_scale
            self.action_histories[i].append(action_magnitude)
        
        self.step_count += 1
        
        # 检查是否所有agent都完成
        all_done = all(self.agent_done)
        
        # 检查超时
        if self.step_count >= self.max_steps:
            for i in range(self.num_agents):
                if not self.agent_done[i]:
                    rewards[i] += self.timeout_penalty
                    self.agent_done[i] = True
            all_done = True
        
        # 如果所有agent都成功，给额外奖励
        if all(self.agent_success):
            for i in range(self.num_agents):
                rewards[i] += self.all_success_bonus / self.num_agents
        
        self.episode_done = all_done
        
        # 获取新观测
        obs_list = self.get_all_obs()
        
        # 汇总信息
        global_info = {
            'step': self.step_count,
            'all_done': all_done,
            'all_success': all(self.agent_success),
            'num_success': sum(self.agent_success),
            'num_collision': sum(self.agent_collided),
            'agent_infos': infos
        }
        
        return obs_list, rewards, self.agent_done.copy(), global_info
    
    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.episode_done = False
        
        # 重新生成障碍物（如果是随机模式）
        if self.use_random_obstacles:
            self.obstacles = self._generate_random_static_obstacles()
        
        # 重新生成动态障碍物
        self.dynamic_obstacles = self._generate_dynamic_obstacles()
        
        # 重新初始化agent
        self._init_agents()
        
        return self.get_all_obs()
    
    def render(self):
        """渲染环境（可选实现）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


# 兼容旧接口
DroneGridEnvironment = MultiAgentDroneEnv


if __name__ == '__main__':
    # 测试环境
    Config.print_obs_dims()
    
    env = MultiAgentDroneEnv()
    obs_list = env.reset()
    
    print(f"\n环境测试:")
    print(f"Agent数量: {env.num_agents}")
    print(f"观测维度: {len(obs_list[0])}")
    print(f"预期观测维度: {Config.get_single_obs_dim()}")
    
    # 测试一步
    actions = [np.random.uniform(-1, 1, 3) for _ in range(env.num_agents)]
    obs_list, rewards, dones, info = env.step(actions)
    
    print(f"\n第一步后:")
    print(f"奖励: {rewards}")
    print(f"完成状态: {dones}")
    print(f"信息: {info}")
