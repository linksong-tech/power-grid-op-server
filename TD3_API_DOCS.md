# TD3强化学习优化 API 文档

## 基础信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`

## 接口列表

### 训练相关接口

#### 1. 开始训练

**接口**: `POST /api/td3-optimize/train`

**描述**: 启动TD3模型训练任务（异步执行）

**请求体**:
```json
{
  "busData": [[1, 0, 0], [2, -2.0, -1.0], [3, 1.5, 0.8]],
  "branchData": [[1, 1, 2, 0.05, 0.15], [2, 2, 3, 0.08, 0.20]],
  "voltageLimits": [9.5, 10.5],
  "keyNodes": [0, 1, 2],
  "tunableNodes": [[1, -2.0, 2.0, "光伏节点2"]],
  "parameters": {
    "UB": 10.38,
    "SB": 10,
    "max_episodes": 800,
    "max_steps": 3,
    "model_name": "my_td3_model"
  }
}
```

**响应示例**:
```json
{
  "status": "success",
  "message": "训练任务已启动",
  "data": {
    "model_name": "my_td3_model",
    "total_episodes": 800
  }
}
```

---

#### 2. 获取训练状态

**接口**: `GET /api/td3-optimize/training-status`

**描述**: 获取当前训练任务的实时状态

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "is_training": true,
    "current_episode": 150,
    "total_episodes": 800,
    "current_reward": 45.6,
    "current_loss_rate": 4.23,
    "best_loss_rate": 3.89,
    "message": "训练中: Episode 150/800",
    "start_time": "2024-12-29T23:00:00"
  }
}
```

---

#### 3. 停止训练

**接口**: `POST /api/td3-optimize/stop-training`

**描述**: 停止当前正在进行的训练任务

**响应示例**:
```json
{
  "status": "success",
  "message": "训练停止请求已发送"
}
```

---

#### 4. 上传训练样本

**接口**: `POST /api/td3-optimize/upload-samples`

**描述**: 上传训练样本数据集

**请求体**:
```json
{
  "samples": [
    {
      "time": "2024-12-29 08:00",
      "ub": 10.0,
      "bus": [1, 0, 0, 2, -2.0, -1.0, 3, 1.5, 0.8]
    },
    {
      "time": "2024-12-29 12:00",
      "ub": 10.0,
      "bus": [1, 0, 0, 2, -2.5, -1.2, 3, 1.8, 1.0]
    }
  ],
  "name": "sample_set_20241229"
}
```

**响应示例**:
```json
{
  "status": "success",
  "message": "训练样本上传成功",
  "data": {
    "sample_name": "sample_set_20241229",
    "sample_count": 2,
    "file_path": "training_samples/sample_set_20241229.json"
  }
}
```

---

#### 5. 获取训练样本列表

**接口**: `GET /api/td3-optimize/samples`

**描述**: 获取已上传的训练样本列表

**响应示例**:
```json
{
  "status": "success",
  "data": [
    {
      "filename": "sample_set_20241229.json",
      "name": "sample_set_20241229",
      "upload_time": "2024-12-29T23:00:00",
      "sample_count": 2
    }
  ]
}
```

---

#### 6. 获取样本详情

**接口**: `GET /api/td3-optimize/samples/<filename>`

**描述**: 获取指定训练样本的详细数据

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "name": "sample_set_20241229",
    "upload_time": "2024-12-29T23:00:00",
    "sample_count": 2,
    "samples": [...]
  }
}
```

---

### 智能体管理接口

#### 7. 获取历史智能体列表

**接口**: `GET /api/td3-optimize/agents`

**描述**: 获取所有训练好的智能体（模型）列表

**响应示例**:
```json
{
  "status": "success",
  "data": [
    {
      "filename": "my_td3_model.pth",
      "model_name": "my_td3_model",
      "size": 1048576,
      "size_mb": 1.0,
      "created_time": "2024-12-29 23:00:00",
      "modified_time": "2024-12-29 23:30:00",
      "training_history": {
        "best_loss_rate": 3.45,
        "total_episodes": 800
      },
      "location": "models"
    }
  ]
}
```

---

#### 8. 获取智能体详情

**接口**: `GET /api/td3-optimize/agents/<model_name>`

**描述**: 获取指定智能体的详细信息和训练历史

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "model_name": "my_td3_model",
    "filename": "my_td3_model.pth",
    "size": 1048576,
    "size_mb": 1.0,
    "created_time": "2024-12-29 23:00:00",
    "modified_time": "2024-12-29 23:30:00",
    "training_history": {
      "rewards": [...],
      "loss_rates": [...]
    },
    "meta_info": {
      "state_dim": 5,
      "action_dim": 2
    }
  }
}
```

---

#### 9. 删除智能体

**接口**: `DELETE /api/td3-optimize/agents/<model_name>`

**描述**: 删除指定的智能体模型

**响应示例**:
```json
{
  "status": "success",
  "message": "智能体删除成功"
}
```

---

### 推理优化接口

#### 10. 执行TD3优化

**接口**: `POST /api/td3-optimize/optimize`

**描述**: 使用训练好的TD3模型对单个电网断面进行无功优化

**请求体**:
```json
{
  "busData": [
    [1, 0, 0],
    [2, -2.0, -1.0],
    [3, 1.5, 0.8]
  ],
  "branchData": [
    [1, 1, 2, 0.05, 0.15],
    [2, 2, 3, 0.08, 0.20]
  ],
  "voltageLimits": [9.5, 10.5],
  "keyNodes": [0, 1, 2],
  "tunableNodes": [
    [1, -2.0, 2.0, "光伏节点2"]
  ],
  "modelPath": "M1_1229_230322.pth",
  "parameters": {
    "UB": 10.38,
    "SB": 10
  }
}
```

**参数说明**:
- `busData`: 节点数据，格式 `[节点号, 有功P(MW), 无功Q(MVar)]`
- `branchData`: 支路数据，格式 `[线路号, 首节点, 末节点, 电阻R(Ω), 电抗X(Ω)]`
- `voltageLimits`: 电压上下限 `[v_min, v_max]` (kV)
- `keyNodes`: 关键监测节点索引列表（节点索引 = 节点号 - 1）
- `tunableNodes`: 可调无功节点配置 `[节点索引, Q_min, Q_max, 节点名称]`
- `modelPath`: 模型文件名（存放在 `models/` 或 `lib/` 目录）
- `parameters.UB`: 基准电压 (kV)
- `parameters.SB`: 基准功率 (MVA)

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "success": true,
    "optimized_q_values": [1.2345, -0.5678],
    "initial_loss_rate": 5.67,
    "optimized_loss_rate": 3.45,
    "loss_reduction": 39.15,
    "optimized_voltages": [10.0, 10.1, 10.2, ...],
    "voltage_violations": 0,
    "q_adjustments": [
      {
        "node_name": "光伏节点2",
        "node_index": 1,
        "initial_q": -1.0,
        "optimized_q": 1.2345,
        "adjustment": 2.2345,
        "q_limits": [-2.0, 2.0]
      }
    ]
  }
}
```

**错误响应**:
```json
{
  "status": "error",
  "message": "模型文件不存在: M1_1229_230322.pth"
}
```

---

### 11. 批量执行TD3优化

**接口**: `POST /api/td3-optimize/batch`

**描述**: 对多个电网断面批量执行TD3优化

**请求体**:
```json
{
  "testSamples": [
    {
      "time": "2024-12-29 08:00",
      "ub": 10.0,
      "bus": [1, 0, 0, 2, -2.0, -1.0, 3, 1.5, 0.8]
    },
    {
      "time": "2024-12-29 12:00",
      "ub": 10.0,
      "bus": [1, 0, 0, 2, -2.5, -1.2, 3, 1.8, 1.0]
    }
  ],
  "branchData": [[1, 1, 2, 0.05, 0.15], [2, 2, 3, 0.08, 0.20]],
  "voltageLimits": [9.5, 10.5],
  "keyNodes": [0, 1, 2],
  "tunableNodes": [[1, -2.0, 2.0, "光伏节点2"]],
  "modelPath": "M1_1229_230322.pth",
  "parameters": {
    "SB": 10
  }
}
```

**参数说明**:
- `testSamples`: 测试样本数组
  - `time`: 时间标识
  - `ub`: 该断面的基准电压 (kV)
  - `bus`: 节点数据（一维数组，格式：节点号1, P1, Q1, 节点号2, P2, Q2, ...）
- 其他参数同单次优化接口

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "success": true,
    "total_samples": 2,
    "successful_optimizations": 2,
    "results": [
      {
        "sample_time": "2024-12-29 08:00",
        "success": true,
        "initial_loss_rate": 5.67,
        "optimized_loss_rate": 3.45,
        "loss_reduction": 39.15,
        "voltage_violations": 0,
        "optimized_q_values": [1.2345],
        "q_adjustments": [...]
      },
      {
        "sample_time": "2024-12-29 12:00",
        "success": true,
        "initial_loss_rate": 6.12,
        "optimized_loss_rate": 3.89,
        "loss_reduction": 36.44,
        "voltage_violations": 0,
        "optimized_q_values": [1.4567],
        "q_adjustments": [...]
      }
    ],
    "summary": {
      "avg_initial_loss": 5.895,
      "avg_optimized_loss": 3.67,
      "avg_loss_reduction": 37.795,
      "avg_voltage_violations": 0
    }
  }
}
```

---

### 12. 获取可用模型列表

**接口**: `GET /api/td3-optimize/models`

**描述**: 获取服务器上可用的TD3模型文件列表

**响应示例**:
```json
{
  "status": "success",
  "data": [
    {
      "filename": "M1_1229_230322.pth",
      "size": 1048576,
      "modified_time": "2024-12-29 23:03:22",
      "location": "lib"
    },
    {
      "filename": "M2_1228_150000.pth",
      "size": 1048576,
      "modified_time": "2024-12-28 15:00:00"
    }
  ]
}
```

---

### 13. 获取TD3优化结果列表

**接口**: `GET /api/td3-optimize/results`

**描述**: 获取历史TD3优化结果列表

**响应示例**:
```json
{
  "status": "success",
  "data": [
    {
      "filename": "td3_result_20241229_230500.json",
      "timestamp": "20241229_230500",
      "initial_loss_rate": 5.67,
      "optimized_loss_rate": 3.45,
      "loss_reduction": 39.15,
      "voltage_violations": 0
    }
  ]
}
```

---

### 14. 获取TD3优化结果详情

**接口**: `GET /api/td3-optimize/results/<filename>`

**描述**: 获取指定优化结果的详细信息

**路径参数**:
- `filename`: 结果文件名，如 `td3_result_20241229_230500.json`

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "success": true,
    "optimized_q_values": [1.2345, -0.5678],
    "initial_loss_rate": 5.67,
    "optimized_loss_rate": 3.45,
    "loss_reduction": 39.15,
    "optimized_voltages": [10.0, 10.1, 10.2, ...],
    "voltage_violations": 0,
    "q_adjustments": [...]
  }
}
```

---

### 15. 获取TD3模板数据

**接口**: `GET /api/td3-optimize/template`

**描述**: 获取TD3优化的示例数据模板，用于测试和参考

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "busData": [[1, 0, 0], [2, -2.0, -1.0], ...],
    "branchData": [[1, 1, 2, 0.05, 0.15], ...],
    "voltageLimits": [9.5, 10.5],
    "keyNodes": [0, 1, 2, 3, 4],
    "tunableNodes": [[32, -0.3, 0.3, "节点33"], [33, -0.5, 0.5, "节点34"]],
    "modelPath": "M1_1229_230322.pth",
    "parameters": {
      "UB": 10.38,
      "SB": 10
    }
  }
}
```

---

## 错误码说明

| HTTP状态码 | 说明 |
|-----------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在（如模型文件、结果文件） |
| 500 | 服务器内部错误 |

## 使用示例

### JavaScript (Fetch API)

```javascript
// 单次优化
async function optimizeWithTD3() {
  const response = await fetch('http://localhost:5000/api/td3-optimize/optimize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      busData: [[1, 0, 0], [2, -2.0, -1.0], [3, 1.5, 0.8]],
      branchData: [[1, 1, 2, 0.05, 0.15], [2, 2, 3, 0.08, 0.20]],
      voltageLimits: [9.5, 10.5],
      keyNodes: [0, 1, 2],
      tunableNodes: [[1, -2.0, 2.0, "光伏节点2"]],
      modelPath: "M1_1229_230322.pth",
      parameters: { UB: 10.38, SB: 10 }
    })
  });
  
  const result = await response.json();
  console.log('优化结果:', result);
}

// 获取模型列表
async function getModels() {
  const response = await fetch('http://localhost:5000/api/td3-optimize/models');
  const result = await response.json();
  console.log('可用模型:', result.data);
}
```

### Python (requests)

```python
import requests

# 单次优化
def optimize_with_td3():
    url = 'http://localhost:5000/api/td3-optimize/optimize'
    data = {
        'busData': [[1, 0, 0], [2, -2.0, -1.0], [3, 1.5, 0.8]],
        'branchData': [[1, 1, 2, 0.05, 0.15], [2, 2, 3, 0.08, 0.20]],
        'voltageLimits': [9.5, 10.5],
        'keyNodes': [0, 1, 2],
        'tunableNodes': [[1, -2.0, 2.0, "光伏节点2"]],
        'modelPath': 'M1_1229_230322.pth',
        'parameters': {'UB': 10.38, 'SB': 10}
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    print('优化结果:', result)

# 批量优化
def batch_optimize():
    url = 'http://localhost:5000/api/td3-optimize/batch'
    data = {
        'testSamples': [
            {'time': '2024-12-29 08:00', 'ub': 10.0, 'bus': [1, 0, 0, 2, -2.0, -1.0, 3, 1.5, 0.8]},
            {'time': '2024-12-29 12:00', 'ub': 10.0, 'bus': [1, 0, 0, 2, -2.5, -1.2, 3, 1.8, 1.0]}
        ],
        'branchData': [[1, 1, 2, 0.05, 0.15], [2, 2, 3, 0.08, 0.20]],
        'voltageLimits': [9.5, 10.5],
        'keyNodes': [0, 1, 2],
        'tunableNodes': [[1, -2.0, 2.0, "光伏节点2"]],
        'modelPath': 'M1_1229_230322.pth',
        'parameters': {'SB': 10}
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    print('批量优化结果:', result)
```

## 注意事项

1. **模型文件位置**: 模型文件应放在 `models/` 目录或 `lib/` 目录下
2. **节点索引**: 节点索引从0开始，节点号从1开始（索引 = 节点号 - 1）
3. **数据单位**: 
   - 功率: MW, MVar
   - 电压: kV
   - 阻抗: Ω
4. **关键节点**: 用于构建状态向量，应选择电网中的关键监测点
5. **可调节点**: 必须是可以调节无功功率的节点（如光伏节点、储能节点等）
6. **模型兼容性**: 使用的模型必须与当前网络结构匹配（state_dim和action_dim一致）

## 性能建议

- 单次优化通常在1-3秒内完成
- 批量优化时间取决于样本数量，建议每批不超过100个样本
- 大规模网络建议使用GPU加速的模型
