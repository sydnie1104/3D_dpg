# 多智能体UAV路径规划 - 多算法实现

本项目实现了三种多智能体强化学习算法用于3D无人机路径规划：

## 📊 算法对比

| 算法 | 类型 | Critic | 训练方式 | 探索策略 | 复杂度 |
|------|------|--------|----------|----------|--------|
| **MADDPG** | Off-policy | 集中式（全局） | 独立更新 | OU噪声 | 高 |
| **IDDPG** | Off-policy | 独立式（局部） | 完全独立 | OU噪声 | 低 |
| **MAPPO** | On-policy | 集中式（全局） | PPO裁剪 | 高斯采样 | 中 |

### 算法特点

- **MADDPG (Multi-Agent DDPG)**
  - ✅ 集中训练，分散执行（CTDE）
  - ✅ Critic看全局状态，Actor只看局部
  - ✅ 适合强协作任务
  - ❌ 训练不稳定，需要仔细调参

- **IDDPG (Independent DDPG)**
  - ✅ 完全独立训练，最简单
  - ✅ 每个agent独立replay buffer
  - ✅ 易于扩展到大规模
  - ❌ 忽略agent间交互，性能较弱

- **MAPPO (Multi-Agent PPO)**
  - ✅ On-policy，训练稳定
  - ✅ 集中式价值函数
  - ✅ PPO裁剪保证单调改进
  - ❌ 样本效率低（需要on-policy数据）

## 🚀 快速开始

### 1. 训练模型

```bash
# 训练MADDPG（原始实现）
python train.py

# 训练IDDPG（独立DDPG）
python train_iddpg.py

# 训练MAPPO（多智能体PPO）
python train_mappo.py
```

### 2. 测试模型

```bash
# 测试MADDPG
python test_multi_algo.py --algo MADDPG

# 测试IDDPG
python test_multi_algo.py --algo IDDPG

# 测试MAPPO
python test_multi_algo.py --algo MAPPO
```

### 3. 可视化结果

```bash
# 可视化训练曲线
python plot_training_metrics.py --algo MADDPG
python plot_training_metrics.py --algo IDDPG
python plot_training_metrics.py --algo MAPPO

# 可视化成功轨迹
python visualize_success.py --algo MADDPG
python visualize_success.py --algo IDDPG
python visualize_success.py --algo MAPPO
```

## 📁 文件结构

```
项目根目录/
├── config.py              # 配置文件（算法类型、超参数）
├── env.py                 # 多智能体环境
│
├── model.py               # MADDPG网络
├── agent.py               # MADDPG智能体
├── train.py               # MADDPG训练
│
├── iddpg_model.py         # IDDPG网络
├── iddpg_agent.py         # IDDPG智能体
├── train_iddpg.py         # IDDPG训练
│
├── mappo_model.py         # MAPPO网络
├── mappo_agent.py         # MAPPO智能体
├── train_mappo.py         # MAPPO训练
│
├── test_multi_algo.py     # 通用测试脚本
├── plot_training_metrics.py  # 训练曲线绘制
├── visualize_success.py   # 轨迹可视化
│
├── logs_maddpg/          # MADDPG日志
├── logs_iddpg/           # IDDPG日志
├── logs_mappo/           # MAPPO日志
├── models_maddpg/        # MADDPG模型
├── models_iddpg/         # IDDPG模型
└── models_mappo/         # MAPPO模型
```

## ⚙️ 配置说明

在 `config.py` 中设置算法类型：

```python
# 算法选择（用于默认路径）
ALGORITHM = "MADDPG"  # 可选: "MADDPG", "MAPPO", "IDDPG"

# 多智能体配置
NUM_AGENTS = 3

# MADDPG/IDDPG参数
ACTOR_LR = 0.0003
CRITIC_LR = 0.001
GAMMA = 0.95
TAU = 0.01
NOISE_SCALE = 0.3

# MAPPO参数
PPO_EPOCH = 10
PPO_CLIP = 0.2
PPO_LR = 0.0003
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
```

## 📈 预期性能

根据任务复杂度，预期性能（全部Agent成功率）：

- **MADDPG**: 40-60%（调参后可达更高）
- **IDDPG**: 20-40%（简单基线）
- **MAPPO**: 30-50%（稳定训练）

## 🔧 调试建议

1. **MADDPG训练不稳定**
   - 降低学习率：`ACTOR_LR=0.0001`, `CRITIC_LR=0.0005`
   - 增大batch size：`BATCH_SIZE=256`
   - 减小噪声：`NOISE_SCALE=0.2`

2. **MAPPO收敛慢**
   - 增加训练轮数：`PPO_EPOCH=15`
   - 调整GAE参数：`GAE_LAMBDA=0.98`
   - 增大熵系数：`ENTROPY_COEF=0.02`

3. **IDDPG性能差**
   - 这是正常的（独立训练忽略交互）
   - 可用作对比基线

## 📊 日志和模型

每个算法的日志和模型按名称分开保存：

- 日志：`logs_{algo}/training_metrics.csv`
- 模型：`models_{algo}/{algo}_model.pth`
- 最佳模型：`models_{algo}/{algo}_model_best.pth`
- 成功模型：`models_{algo}/{algo}_model_allsuccess_ep{N}_{timestamp}.pth`

## 🎯 论文对比实验建议

1. **训练三个算法**，各1000-2000轮
2. **使用相同的随机种子**（如果需要）
3. **记录指标**：
   - 平均总奖励
   - Agent成功率
   - 全部Agent成功率
   - 平均碰撞次数
   - 平均步数
4. **绘制对比曲线**：
   ```bash
   python plot_training_metrics.py --algo MADDPG
   python plot_training_metrics.py --algo IDDPG
   python plot_training_metrics.py --algo MAPPO
   ```

## 📝 引用

如果使用本项目，请引用相关论文：

- **MADDPG**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017
- **IDDPG**: 基于独立学习的扩展
- **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2021

---

**提示**: 所有算法共享相同的环境和网络架构（Transformer+Ray Casting），差异仅在训练策略。

