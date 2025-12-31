# TD3无功优化服务 API 文档

## 概述

本项目提供了工程化的TD3（Twin Delayed Deep Deterministic Policy Gradient）强化学习服务，用于电力系统无功优化。

## 目录结构

```
lib/
├── td3_core/                    # 核心模块
│   ├── __init__.py
│   ├── models.py               # Actor和Critic网络定义
│   └── power_flow.py           # 潮流计算函数
├── td3_train_service.py        # 训练服务API
├── td3_inference_service.py    # 推理服务API
├── td3_example_usage.py        # 使用示例
└── TD3_API_README.md           # 本文档
```

## 快速开始

### 1. 训练模型

```python
from td3_train_service import train_td3_model
import numpy as np

# 准备数据
bus_data = np.array([...])  # 节点数据 (n×3)
branch_data = np.array([...])  # 支路数据 (m×5)
voltage_limits = (9.5, 10.5)  # 电压上下限 kV
key_nodes = [0, 1, 2]  # 关键节点索引
tunable_q_nodes = [(1, -2.0, 2.0, "节点2")]  # 可调无功节点
ub = 10.0  # 基准电压

# 训练
result = train_td3_model(
    bus_data=bus_data,
    branch_data=branch_data,
    voltage_limits=voltage_limits,
    key_nodes=key_nodes,
    tunable_q_nodes=tunable_q_nodes,
    ub=ub,
    max_episodes=800,
    model_save_path="my_model.pth"
)

print(f"最优网损率: {result['best_loss_rate']:.4f}%")
print(f"模型路径: {result['model_path']}")
```

### 2. 单次优化

```python
from td3_inference_service import optimize_reactive_power

# 执行优化
result = optimize_reactive_power(
    bus_data=bus_data,
    branch_data=branch_data,
    voltage_limits=voltage_limits,
    key_nodes=key_nodes,
    tunable_q_nodes=tunable_q_nodes,
    model_path="my_model.pth",
    ub=ub
)

if result['success']:
    print(f"优化前网损率: {result['initial_loss_rate']:.4f}%")
    print(f"优化后网损率: {result['optimized_loss_rate']:.4f}%")
    print(f"网损降低: {result['loss_reduction']:.2f}%")
```

### 3. 批量优化

```python
from td3_inference_service import batch_optimize

test_samples = [
    {'time': '2024-12-29 08:00', 'ub': 10.0, 'bus': np.array([...])},
    {'time': '2024-12-29 12:00', 'ub': 10.0, 'bus': np.array([...])},
]

result = batch_optimize(
    test_samples=test_samples,
    branch_data=branch_data,
    voltage_limits=voltage_limits,
    key_nodes=key_nodes,
    tunable_q_nodes=tunable_q_nodes,
    model_path="my_model.pth"
)

print(f"成功优化: {result['successful_optimizations']}/{result['total_samples']}")
print(f"平均网损降低: {result['summary']['avg_loss_reduction']:.2f}%")
```

## API 参考

### 训练服务

#### `train_td3_model()`

训练TD3模型的主函数。

**参数:**
- `bus_data` (np.ndarray): 节点数据 (n×3) [节点号, 有功P, 无功Q]
- `branch_data` (np.ndarray): 支路数据 (m×5) [线路号, 首节点, 末节点, 电阻R, 电抗X]
- `voltage_limits` (Tuple[float, float]): 电压上下限 (v_min, v_max) kV
- `key_nodes` (List[int]): 关键节点索引列表
- `tunable_q_nodes` (List[Tuple]): 可调无功节点配置 [(节点索引, Q_min, Q_max, 节点名), ...]
- `ub` (float): 基准电压 kV
- `sb` (float): 基准功率 MVA，默认10.0
- `max_episodes` (int): 最大训练轮次，默认800
- `max_steps` (int): 每轮最大步数，默认3
- `model_save_path` (str): 模型保存路径
- `progress_callback` (Callable): 进度回调函数，可选

**返回:**
```python
{
    'success': bool,
    'best_loss_rate': float,
    'model_path': str,
    'training_history': {
        'rewards': List[float],
        'loss_rates': List[float]
    }
}
```

### 推理服务

#### `optimize_reactive_power()`

使用TD3模型优化无功功率配置。

**参数:**
- `bus_data` (np.ndarray): 节点数据 (n×3)
- `branch_data` (np.ndarray): 支路数据 (m×5)
- `voltage_limits` (Tuple[float, float]): 电压上下限 kV
- `key_nodes` (List[int]): 关键节点索引列表
- `tunable_q_nodes` (List[Tuple]): 可调无功节点配置
- `model_path` (str): 训练好的模型文件路径
- `ub` (float): 基准电压 kV
- `sb` (float): 基准功率 MVA，默认10.0

**返回:**
```python
{
    'success': bool,
    'optimized_q_values': List[float],  # 优化后的无功值
    'initial_loss_rate': float,  # 优化前网损率
    'optimized_loss_rate': float,  # 优化后网损率
    'loss_reduction': float,  # 网损降低百分比
    'optimized_voltages': np.ndarray,  # 优化后节点电压
    'voltage_violations': int,  # 电压越限节点数
    'q_adjustments': List[Dict]  # 无功调节详情
}
```

#### `batch_optimize()`

批量优化多个测试样本。

**参数:**
- `test_samples` (List[Dict]): 测试样本列表，每个样本格式:
  ```python
  {
      'time': str,  # 时间标识
      'ub': float,  # 基准电压
      'bus': np.ndarray  # 节点数据（一维数组）
  }
  ```
- `branch_data` (np.ndarray): 支路数据
- `voltage_limits` (Tuple[float, float]): 电压上下限
- `key_nodes` (List[int]): 关键节点索引
- `tunable_q_nodes` (List[Tuple]): 可调无功节点配置
- `model_path` (str): 模型路径
- `sb` (float): 基准功率，默认10.0

**返回:**
```python
{
    'success': bool,
    'total_samples': int,
    'successful_optimizations': int,
    'results': List[Dict],  # 每个样本的优化结果
    'summary': {
        'avg_initial_loss': float,
        'avg_optimized_loss': float,
        'avg_loss_reduction': float,
        'avg_voltage_violations': float
    }
}
```

## 数据格式说明

### 节点数据 (bus_data)
形状: (n, 3)，n为节点数
```
[节点号, 有功P(MW), 无功Q(MVar)]
```
示例:
```python
np.array([
    [1, -5.0, 0.0],   # 节点1: 平衡节点
    [2, -2.0, -1.0],  # 节点2: 光伏节点（有功为负）
    [3, 1.5, 0.8],    # 节点3: 负荷节点（有功为正）
])
```

### 支路数据 (branch_data)
形状: (m, 5)，m为支路数
```
[线路号, 首节点, 末节点, 电阻R(Ω), 电抗X(Ω)]
```
示例:
```python
np.array([
    [1, 1, 2, 0.05, 0.15],
    [2, 2, 3, 0.08, 0.20],
])
```

### 可调无功节点配置 (tunable_q_nodes)
格式: List[Tuple[int, float, float, str]]
```
[(节点索引, Q_min, Q_max, 节点名称), ...]
```
示例:
```python
[
    (1, -2.0, 2.0, "光伏节点2"),  # 节点索引1，无功范围[-2, 2] MVar
    (3, -1.5, 1.5, "光伏节点4"),
]
```

## 注意事项

1. **节点索引**: 节点索引从0开始，节点号从1开始。节点索引 = 节点号 - 1
2. **单位**: 
   - 功率: MW, MVar
   - 电压: kV
   - 阻抗: Ω
3. **平衡节点**: 必须是节点号为1的节点（索引0）
4. **模型兼容性**: 推理时使用的网络结构（state_dim, action_dim）必须与训练时一致

## 性能优化建议

1. **训练参数调优**:
   - `max_episodes`: 根据收敛情况调整，通常500-1000轮
   - `max_steps`: 每轮步数，建议3-5步
   - 探索噪声会自动衰减

2. **GPU加速**: 
   - 自动检测并使用GPU（如果可用）
   - 大规模网络建议使用GPU训练

3. **经验回放**:
   - 默认缓冲区大小50000
   - 预填充5000条经验后开始训练

## 故障排查

### 潮流计算不收敛
- 检查网络拓扑是否正确
- 检查功率数据是否合理
- 调整收敛精度参数 `pr`

### 训练不稳定
- 降低学习率
- 增加预填充经验数量
- 检查奖励函数设计

### 优化效果不佳
- 增加训练轮次
- 调整关键节点选择
- 检查电压约束设置

## 示例代码

完整示例请参考 `td3_example_usage.py`

## 联系方式

如有问题，请联系开发团队。
