# environment_renderer.py - 多智能体环境渲染器
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from config import Config

plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# Agent颜色列表
AGENT_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']


def to_xyz(pos):
    """通用坐标提取函数"""
    if pos is None:
        return 0.0, 0.0, 0.0
    if hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z"):
        return pos.x, pos.y, pos.z
    arr = np.array(pos, dtype=float).flatten()
    if arr.size >= 3:
        return arr[0], arr[1], arr[2]
    elif arr.size == 2:
        return arr[0], arr[1], 0.0
    elif arr.size == 1:
        return arr[0], 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0


class MultiAgentRenderer:
    """多智能体环境渲染器"""
    
    def __init__(self, grid_scale, num_agents=None):
        self.grid_scale = grid_scale
        self.num_agents = num_agents or Config.NUM_AGENTS
        self.render_delay = getattr(Config, 'RENDER_DELAY', 0.05)
        
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        self.fig.show()
    
    def render(self, obstacles, dynamic_obstacles, drone_positions, target_positions,
               path_histories, step_count, agent_done=None, additional_info=None):
        """
        渲染多智能体环境
        
        Args:
            obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            drone_positions: List[Point], 每个agent的位置
            target_positions: List[Point], 每个agent的目标位置
            path_histories: List[List[Point]], 每个agent的路径历史
            step_count: 当前步数
            agent_done: List[bool], 每个agent是否完成
            additional_info: 附加信息字符串
        """
        self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlabel('X轴', fontsize=12)
        self.ax.set_ylabel('Y轴', fontsize=12)
        self.ax.set_zlabel('Z轴', fontsize=12)
        self.ax.set_xlim((0, self.grid_scale))
        self.ax.set_ylim((0, self.grid_scale))
        self.ax.set_zlim((0, self.grid_scale))
        
        # 标题
        title = f'多智能体路径规划 (步数: {step_count}, Agent数: {self.num_agents})'
        if additional_info:
            title += f' - {additional_info}'
        self.ax.set_title(title, fontsize=14)
        
        # 绘制静态障碍物
        if obstacles:
            for obstacle in obstacles:
                if isinstance(obstacle, tuple) and len(obstacle) >= 5:
                    x, y, bd_w, bd_l, h = obstacle
                else:
                    continue
                self._draw_box(x, y, bd_w, bd_l, h)
        
        # 绘制动态障碍物
        if dynamic_obstacles:
            for obstacle in dynamic_obstacles:
                if hasattr(obstacle, 'pos'):
                    pos = obstacle.pos
                    radius = getattr(obstacle, 'radius', 1.0)
                elif isinstance(obstacle, (tuple, list, np.ndarray)):
                    pos = obstacle
                    radius = 1.0
                else:
                    continue
                
                x0, y0, z0 = to_xyz(pos)
                self._draw_sphere(x0, y0, z0, radius, color='red', alpha=0.6)
        
        # 绘制每个agent的路径、位置和目标
        for i in range(min(len(drone_positions), self.num_agents)):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            is_done = agent_done[i] if agent_done else False
            
            # 绘制路径
            if i < len(path_histories) and path_histories[i]:
                path_points = [to_xyz(p) for p in path_histories[i]]
                path_x, path_y, path_z = zip(*path_points)
                self.ax.plot(path_x, path_y, path_z, color=color, alpha=0.7, 
                           linewidth=2, label=f'Agent {i} 路径')
            
            # 绘制当前位置
            if i < len(drone_positions) and drone_positions[i] is not None:
                x, y, z = to_xyz(drone_positions[i])
                marker = 'o' if is_done else '^'
                self.ax.scatter(x, y, z, color=color, s=200, marker=marker, 
                              edgecolors='black', linewidths=2)
            
            # 绘制目标位置
            if i < len(target_positions) and target_positions[i] is not None:
                x, y, z = to_xyz(target_positions[i])
                self.ax.scatter(x, y, z, color=color, s=300, marker='*', 
                              edgecolors='black', linewidths=2, alpha=0.8)
        
        # 图例
        self.ax.legend(loc='upper right', fontsize=8)
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if self.render_delay > 0:
            time.sleep(self.render_delay)
    
    def _draw_box(self, x, y, bd_w, bd_l, h, color='gray', alpha=0.6):
        """绘制长方体障碍物"""
        x_grid = np.linspace(x - bd_w, x + bd_w, 5)
        y_grid = np.linspace(y - bd_l, y + bd_l, 5)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        
        # 顶面和底面
        for z_val in [0, h]:
            self.ax.plot_surface(x_mesh, y_mesh, np.full_like(x_mesh, z_val),
                               rstride=1, cstride=1, color=color, alpha=alpha)
        
        # 四个侧面
        y_side = np.linspace(y - bd_l, y + bd_l, 5)
        z_side = np.linspace(0, h, 5)
        y_mesh_side, z_mesh_side = np.meshgrid(y_side, z_side)
        for x_val in [x - bd_w, x + bd_w]:
            self.ax.plot_surface(np.full_like(y_mesh_side, x_val),
                               y_mesh_side, z_mesh_side,
                               color=color, alpha=alpha)
        
        x_front = np.linspace(x - bd_w, x + bd_w, 5)
        z_front = np.linspace(0, h, 5)
        x_mesh_front, z_mesh_front = np.meshgrid(x_front, z_front)
        for y_val in [y - bd_l, y + bd_l]:
            self.ax.plot_surface(x_mesh_front,
                               np.full_like(x_mesh_front, y_val),
                               z_mesh_front,
                               color=color, alpha=alpha)
    
    def _draw_sphere(self, x0, y0, z0, radius, color='red', alpha=0.8):
        """绘制球体"""
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x_sphere = x0 + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = y0 + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = z0 + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x_sphere, y_sphere, z_sphere,
                           rstride=1, cstride=1, color=color, alpha=alpha)
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)


# 兼容旧接口
EnvironmentRenderer = MultiAgentRenderer


if __name__ == '__main__':
    # 测试渲染器
    from env import Point
    
    renderer = MultiAgentRenderer(grid_scale=20, num_agents=3)
    
    # 模拟数据
    obstacles = [(10, 10, 0.5, 0.5, 10)]
    dynamic_obstacles = []
    
    drone_positions = [Point(2, 2, 2), Point(2, 18, 2), Point(18, 2, 2)]
    target_positions = [Point(18, 18, 18), Point(18, 2, 18), Point(2, 18, 18)]
    path_histories = [[p] for p in drone_positions]
    
    renderer.render(obstacles, dynamic_obstacles, drone_positions, target_positions,
                   path_histories, step_count=0, agent_done=[False, False, False])
    
    plt.show()
    input("按Enter关闭...")
    renderer.close()
