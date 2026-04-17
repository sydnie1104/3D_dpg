class Config:
    """多智能体算法及环境配置"""
    
    # ========== 算法类型配置 ==========
    ALGORITHM = "MADDPG"  # 可选: "MADDPG", "MAPPO", "IDDPG"
    
    # ========== 多智能体配置 ==========
    NUM_AGENTS = 3  # UAV数量（可配置）
    
    # 每个UAV的起点和终点（None表示随机生成）
    # 格式：[(x, y, z), ...] 或 None
    # 固定起点终点（NUM_AGENTS=3时）- 左下角区域→右上角区域（避开障碍物）
    AGENT_START_POSITIONS = [
        (2.0, 2.0, 2.0),   # Agent 0: 左下角（安全）
        (3.5, 1.5, 3.0),   # Agent 1: 左下角区域（避开5,5障碍物）
        (1.5, 3.5, 3.5)    # Agent 2: 左下角区域（避开5,5障碍物）
    ]
    AGENT_TARGET_POSITIONS = [
        (18.0, 18.0, 18.0),  # Agent 0: 右上角
        (17.0, 18.5, 17.0),  # Agent 1: 右上角区域（较大错开）
        (18.5, 17.0, 17.5)   # Agent 2: 右上角区域（较大错开）
    ]
    
    # Agent之间的安全距离
    AGENT_COLLISION_DIST = 1.0  # agent之间碰撞判定距离
    AGENT_SAFE_DIST = 2.0  # agent之间安全距离（用于奖励计算）
    
    # ========== 观测空间配置（动态计算维度）==========
    # 自身信息
    SELF_POS_DIM = 3  # 自身位置 (x, y, z)
    SELF_VEL_DIM = 3  # 自身速度 (vx, vy, vz)
    GOAL_REL_DIM = 3  # 相对目标位置 (dx, dy, dz)
    GOAL_DIST_DIM = 1  # 到目标距离
    
    # Ray Casting配置
    NUM_RAY_DIRECTIONS = 26  # 射线方向数（6主轴 + 12面对角 + 8体对角）
    RAY_INFO_PER_DIR = 2  # 每个射线的信息维度（距离 + 类型）
    
    # 局部目标向量
    LOCAL_GOAL_DIM = 5  # 目标可见性 + 夹角 + 障碍距离 + 俯仰角 + 方位角
    
    # 动态障碍物
    NUM_DYNAMIC_OBSTACLES = 5  # 动态障碍物数量
    DYN_OBS_INFO_DIM = 7  # 每个动态障碍物信息维度（位置3 + 半径1 + 速度3）
    
    # 动作历史
    ACTION_HISTORY_LEN = 5  # 动作历史长度
    
    # 其他Agent信息
    OTHER_AGENT_INFO_DIM = 7  # 每个其他agent的信息维度（相对位置3 + 相对速度3 + 距离1）
    
    @classmethod
    def get_self_obs_dim(cls):
        """计算单个agent自身观测维度（不含其他agent信息）"""
        return (
            cls.GOAL_REL_DIM +  # 相对目标
            cls.GOAL_DIST_DIM +  # 目标距离
            cls.SELF_VEL_DIM +  # 自身速度
            cls.NUM_RAY_DIRECTIONS * cls.RAY_INFO_PER_DIR +  # Ray Casting
            cls.LOCAL_GOAL_DIM +  # 局部目标向量
            cls.NUM_DYNAMIC_OBSTACLES * cls.DYN_OBS_INFO_DIM +  # 动态障碍物
            cls.ACTION_HISTORY_LEN  # 动作历史
        )
    
    @classmethod
    def get_other_agents_obs_dim(cls):
        """计算其他agent信息的观测维度"""
        return (cls.NUM_AGENTS - 1) * cls.OTHER_AGENT_INFO_DIM
    
    @classmethod
    def get_single_obs_dim(cls):
        """计算单个agent的完整观测维度"""
        return cls.get_self_obs_dim() + cls.get_other_agents_obs_dim()
    
    @classmethod
    def get_global_state_dim(cls):
        """计算全局状态维度（用于Critic）"""
        return cls.get_single_obs_dim() * cls.NUM_AGENTS
    
    @classmethod
    def get_action_dim(cls):
        """获取动作维度"""
        return 3  # (vx, vy, vz) 连续动作
    
    @classmethod
    def print_obs_dims(cls):
        """打印观测空间维度信息"""
        print("=" * 60)
        print("观测空间维度配置")
        print("=" * 60)
        print(f"智能体数量: {cls.NUM_AGENTS}")
        print(f"\n单Agent自身观测维度分解:")
        print(f"  - 相对目标位置: {cls.GOAL_REL_DIM}")
        print(f"  - 目标距离: {cls.GOAL_DIST_DIM}")
        print(f"  - 自身速度: {cls.SELF_VEL_DIM}")
        print(f"  - Ray Casting: {cls.NUM_RAY_DIRECTIONS} × {cls.RAY_INFO_PER_DIR} = {cls.NUM_RAY_DIRECTIONS * cls.RAY_INFO_PER_DIR}")
        print(f"  - 局部目标向量: {cls.LOCAL_GOAL_DIM}")
        print(f"  - 动态障碍物: {cls.NUM_DYNAMIC_OBSTACLES} × {cls.DYN_OBS_INFO_DIM} = {cls.NUM_DYNAMIC_OBSTACLES * cls.DYN_OBS_INFO_DIM}")
        print(f"  - 动作历史: {cls.ACTION_HISTORY_LEN}")
        print(f"  = 自身观测小计: {cls.get_self_obs_dim()}")
        print(f"\n其他Agent观测维度:")
        print(f"  - 其他Agent数: {cls.NUM_AGENTS - 1}")
        print(f"  - 每个Agent信息: {cls.OTHER_AGENT_INFO_DIM}")
        print(f"  = 其他Agent小计: {cls.get_other_agents_obs_dim()}")
        print(f"\n单Agent完整观测维度: {cls.get_single_obs_dim()}")
        print(f"全局状态维度(Critic): {cls.get_global_state_dim()}")
        print(f"动作维度: {cls.get_action_dim()}")
        print("=" * 60)
    
    # ========== 环境参数 ==========
    GRID_SCALE = 20  # 空间范围（0到GRID_SCALE）
    MAX_STEPS = 150  # 每轮最大步数
    D_SAFE = 1.0  # 到达目标的安全距离
    ACTION_SCALE = 0.6  # 动作缩放因子
    
    # ========== 障碍物模式配置 ==========
    USE_RANDOM_OBSTACLES = True  # True=随机生成障碍物, False=使用固定障碍物
    
    # ========== 训练参数 ==========
    TOTAL_EPISODES = 2000  # 总训练轮次
    BATCH_SIZE = 128  # 批处理大小
    REPLAY_BUFFER_SIZE = 100000  # 经验回放缓冲区大小
    
    # ========== 路径配置（根据算法动态生成）==========
    @staticmethod
    def get_log_dir():
        """根据算法类型和障碍物模式返回日志目录"""
        algo = Config.ALGORITHM.lower()
        obstacle_suffix = "" if Config.USE_RANDOM_OBSTACLES else "_fixed"
        return f"logs_{algo}{obstacle_suffix}"
    
    @staticmethod
    def get_model_dir():
        """根据算法类型和障碍物模式返回模型目录"""
        algo = Config.ALGORITHM.lower()
        obstacle_suffix = "" if Config.USE_RANDOM_OBSTACLES else "_fixed"
        return f"models_{algo}{obstacle_suffix}"
    
    @staticmethod
    def get_model_name():
        """返回模型名称"""
        return f"{Config.ALGORITHM.lower()}_model"
    
    # 兼容旧代码
    LOG_DIR = "logs"
    MODEL_DIR = "models"
    MODEL_NAME = "maddpg_model"
    
    # ========== 网络参数（MADDPG / IDDPG）==========
    ACTOR_LR = 0.0003  # Actor学习率
    CRITIC_LR = 0.001  # Critic学习率
    GAMMA = 0.95  # 折扣因子
    TAU = 0.01  # 软更新系数（DDPG系列）
    
    # ========== MAPPO特有参数 ==========
    PPO_EPOCH = 10  # PPO更新轮数
    PPO_CLIP = 0.2  # PPO裁剪参数
    PPO_LR = 0.0003  # PPO学习率
    VALUE_COEF = 0.5  # 价值函数损失系数
    ENTROPY_COEF = 0.01  # 熵正则化系数
    GAE_LAMBDA = 0.95  # GAE λ参数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪
    
    # ========== 探索噪声参数（DDPG系列）==========
    NOISE_SCALE = 0.3  # 初始噪声强度
    NOISE_DECAY = 0.9995  # 噪声衰减因子
    NOISE_MIN = 0.05  # 最小噪声
    
    # ========== 可视化参数 ==========
    RENDER_TRAIN = False  # 训练时是否可视化
    RENDER_TEST = True  # 测试时是否可视化
    RENDER_DELAY = 0.05  # 渲染延迟
    
    # ========== 奖励参数 ==========
    COLLISION_PENALTY = -100  # 碰撞惩罚（降低单次惩罚，因为多agent）
    SUCCESS_REWARD = 200  # 单个agent到达目标奖励
    ALL_SUCCESS_BONUS = 500  # 所有agent都到达的额外奖励
    TIMEOUT_PENALTY = -50  # 超时惩罚
    AGENT_COLLISION_PENALTY = -80  # agent之间碰撞惩罚
    CROSS_BORDER_PENALTY = -100  # 越界惩罚
    
    # ========== 动态障碍物参数 ==========
    DYNAMIC_OBSTACLE_MIN_RADIUS = 0.4
    DYNAMIC_OBSTACLE_MAX_RADIUS = 0.8
    DYNAMIC_OBSTACLE_MIN_SPEED = 0.3
    DYNAMIC_OBSTACLE_MAX_SPEED = 0.8
    CUSTOM_DYNAMIC_OBSTACLES = []
    
    # ========== 固定障碍物配置 ==========
    FIXED_OBSTACLES = [
        # 格式：(x, y, 半宽, 半长, 高度)
        # 中部区域（8个）
        (7, 7, 0.4, 0.4, 6),
        (9, 7, 0.4, 0.4, 5),
        (11, 7, 0.4, 0.4, 6),
        (13, 7, 0.4, 0.4, 5),
        (7, 13, 0.4, 0.4, 5),
        (9, 13, 0.4, 0.4, 6),
        (11, 13, 0.4, 0.4, 5),
        (13, 13, 0.4, 0.4, 6),
        # 边缘区域（8个）
        (5, 5, 0.4, 0.4, 8),
        (5, 10, 0.4, 0.4, 7),
        (5, 15, 0.4, 0.4, 8),
        (15, 5, 0.4, 0.4, 8),
        (15, 10, 0.4, 0.4, 7),
        (15, 15, 0.4, 0.4, 8),
        (10, 5, 0.4, 0.4, 7),
        (10, 15, 0.4, 0.4, 7),
        # 对角线分布（5个）
        (8, 9, 0.4, 0.4, 9),
        (10, 10, 0.4, 0.4, 10),
        (12, 11, 0.4, 0.4, 9),
        (9, 11, 0.4, 0.4, 8),
        (11, 9, 0.4, 0.4, 8),
        # 额外分散点（4个）
        (6, 12, 0.4, 0.4, 5),
        (14, 8, 0.4, 0.4, 5),
        (8, 15, 0.4, 0.4, 6),
        (12, 5, 0.4, 0.4, 6)
    ]


# 测试维度计算
if __name__ == '__main__':
    Config.print_obs_dims()
