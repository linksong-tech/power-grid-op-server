# 电力潮流计算服务

基于前推回代法的配电网潮流计算后端服务，提供RESTful API接口支持前端应用调用。

## 功能特性

- ✅ **前推回代法潮流计算**：适用于辐射状配电网
- ✅ **PSO无功优化**：基于粒子群算法的无功功率优化
- ✅ **RESTful API接口**：支持参数配置、计算执行、结果查询
- ✅ **跨域支持**：支持前端跨域请求
- ✅ **数据持久化**：自动保存计算参数和结果
- ✅ **模板数据**：提供示例数据模板
- ✅ **结果导出**：支持计算结果JSON格式导出
- ✅ **健康检查**：服务状态监控

## 技术栈

- **后端框架**：Flask 3.0.0
- **跨域支持**：Flask-CORS 4.0.0
- **数值计算**：NumPy 1.24.3
- **数据格式**：JSON
- **Python版本**：3.8+

## 快速开始

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 或者使用启动脚本自动安装
python start_server.py --install-deps
```

### 2. 启动服务

```bash
# 方式1：直接启动
python app.py

# 方式2：使用启动脚本（推荐）
python start_server.py

# 方式3：自定义配置启动
python start_server.py --host 0.0.0.0 --port 5000 --no-debug
```

### 3. 验证服务

访问 `http://localhost:5000/api/health` 查看服务状态。

## API接口文档

### 基础信息

- **服务地址**：`http://localhost:5000`
- **数据格式**：JSON
- **字符编码**：UTF-8

### 接口列表

#### 1. 健康检查

```http
GET /api/health
```

**响应示例：**
```json
{
  "status": "success",
  "message": "电力潮流计算服务运行正常",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. 保存计算参数

```http
POST /api/flow-compute/parameters
Content-Type: application/json

{
  "baseVoltage": "10.3",
  "basePower": "10.38", 
  "convergencePrecision": "1e-6"
}
```

#### 3. 获取计算参数

```http
GET /api/flow-compute/parameters
```

#### 4. 执行潮流计算

```http
POST /api/flow-compute/calculate
Content-Type: application/json

{
  "baseVoltage": 10.3,
  "basePower": 10.38,
  "convergencePrecision": 1e-6,
  "busData": [
    [1, 0, 0],
    [2, 0.5, 0.1],
    [3, -0.2, 0]
  ],
  "branchData": [
    [1, 1, 2, 0.1, 0.2],
    [2, 2, 3, 0.05, 0.1]
  ]
}
```

**响应示例：**
```json
{
  "status": "success",
  "parameters": {
    "base_power_mva": 10.38,
    "base_voltage_kv": 10.3,
    "precision": 1e-6
  },
  "node_results": [
    {
      "node_id": 1,
      "node_type": "平衡节点",
      "voltage_kv": 10.3,
      "angle_deg": 0.0,
      "load_p_mw": 0.0,
      "load_q_mvar": 0.0
    }
  ],
  "branch_results": [
    {
      "branch_id": 1,
      "from_node": 1,
      "to_node": 2,
      "active_power_mw": 0.5,
      "reactive_power_mvar": 0.1,
      "active_loss_mw": 0.001,
      "reactive_loss_mvar": 0.002
    }
  ],
  "summary": {
    "total_active_loss_mw": 0.001,
    "total_reactive_loss_mvar": 0.002,
    "balance_node_output_mw": 0.5,
    "pv_total_injection_mw": 0.2,
    "total_input_power_mw": 0.7,
    "total_output_power_mw": 0.5,
    "loss_rate_percent": 28.57,
    "iteration_count": 5,
    "converged": true
  }
}
```

#### 5. 获取示例模板

```http
GET /api/flow-compute/template
```

#### 6. 获取结果列表

```http
GET /api/flow-compute/results
```

#### 7. 获取结果详情

```http
GET /api/flow-compute/results/{filename}
```

#### 8. 导出结果

```http
GET /api/flow-compute/export/{filename}
```

### PSO 无功优化接口

#### 9. 保存 PSO 参数

```http
POST /api/pso-optimize/parameters
Content-Type: application/json

{
  "num_particles": 30,
  "max_iter": 50,
  "w": 0.8,
  "c1": 1.5,
  "c2": 1.5,
  "v_min": 0.95,
  "v_max": 1.05,
  "SB": 10,
  "UB": 10.38,
  "pr": 1e-6
}
```

#### 10. 获取 PSO 参数

```http
GET /api/pso-optimize/parameters
```

#### 11. 执行 PSO 无功优化

```http
POST /api/pso-optimize/optimize
Content-Type: application/json

{
  "busData": [
    [1, 0, 0],
    [2, 0, 0],
    [33, -0.6704, 0.0386],
    [34, -1.205, 0.414]
  ],
  "branchData": [
    [1, 1, 2, 0.3436, 0.8136],
    [33, 10, 34, 0.16625, 0.1204]
  ],
  "tunableNodes": [
    [32, -0.3, 0.3, "节点33"],
    [33, -0.5, 0.5, "节点34"]
  ],
  "psoParameters": {
    "num_particles": 30,
    "max_iter": 50,
    "w": 0.8,
    "c1": 1.5,
    "c2": 1.5,
    "v_min": 0.95,
    "v_max": 1.05
  }
}
```

**响应示例：**
```json
{
  "status": "success",
  "data": {
    "optimal_params": [0.15, -0.2],
    "optimal_loss_rate": 2.45,
    "initial_loss_rate": 3.12,
    "loss_reduction": 0.67,
    "loss_reduction_percent": 21.47,
    "fitness_history": [3.12, 2.89, 2.67, 2.45],
    "initial_q_values": [0.0386, 0.414],
    "node_names": ["节点33", "节点34"],
    "final_voltages": [10.3, 10.2, 10.1, 10.0],
    "initial_voltages": [10.3, 10.1, 9.8, 9.9],
    "convergence": true,
    "power_info": {
      "balance_node_output": 0.5,
      "pv_total_injection": 1.875,
      "total_input_power": 2.375,
      "total_output_power": 2.32
    }
  }
}
```

#### 12. 获取 PSO 模板

```http
GET /api/pso-optimize/template
```

#### 13. 获取 PSO 结果列表

```http
GET /api/pso-optimize/results
```

## 数据格式说明

### 节点数据格式

```json
[
  [节点号, 负荷有功(MW), 负荷无功(MVar)]
]
```

**节点类型说明：**
- **平衡节点**：节点号为1，作为参考节点
- **负荷节点**：有功功率 > 0，表示用电负荷
- **光伏节点**：有功功率 < 0，表示分布式发电
- **普通节点**：有功功率 = 0，无功率注入/消耗

### 支路数据格式

```json
[
  [支路号, 首节点, 尾节点, 电阻(Ω), 电抗(Ω)]
]
```

### PSO 优化数据格式

#### 可调节点配置

```json
[
  [节点索引, 无功最小值(MVar), 无功最大值(MVar), 节点名称]
]
```

**示例：**
```json
[
  [32, -0.3, 0.3, "节点33"],
  [33, -0.5, 0.5, "节点34"]
]
```

#### PSO 算法参数

```json
{
  "num_particles": 30,    // 粒子数量
  "max_iter": 50,         // 最大迭代次数
  "w": 0.8,               // 初始惯性权重
  "c1": 1.5,              // 认知系数
  "c2": 1.5,              // 社会系数
  "v_min": 0.95,          // 电压下限(标幺值)
  "v_max": 1.05,          // 电压上限(标幺值)
  "SB": 10,               // 基准功率(MVA)
  "UB": 10.38,            // 基准电压(kV)
  "pr": 1e-6              // 潮流收敛精度
}
```

#### PSO 优化结果格式

```json
{
  "optimal_params": [0.15, -0.2],           // 最优无功参数
  "optimal_loss_rate": 2.45,                 // 最优网损率(%)
  "initial_loss_rate": 3.12,                 // 初始网损率(%)
  "loss_reduction": 0.67,                    // 网损率降低值(%)
  "loss_reduction_percent": 21.47,          // 网损率降低百分比(%)
  "fitness_history": [3.12, 2.89, 2.67, 2.45], // 优化过程历史
  "initial_q_values": [0.0386, 0.414],      // 初始无功值
  "node_names": ["节点33", "节点34"],        // 节点名称
  "final_voltages": [10.3, 10.2, 10.1, 10.0], // 最终电压分布
  "initial_voltages": [10.3, 10.1, 9.8, 9.9], // 初始电压分布
  "convergence": true,                       // 收敛状态
  "power_info": {                           // 功率信息
    "balance_node_output": 0.5,
    "pv_total_injection": 1.875,
    "total_input_power": 2.375,
    "total_output_power": 2.32
  }
}
```

## 前端集成示例

### React/TypeScript 集成

```typescript
// API客户端类
class PowerFlowAPI {
  private baseURL = 'http://localhost:5000';
  
  async calculatePowerFlow(params: {
    baseVoltage: number;
    basePower: number;
    convergencePrecision: number;
    busData: number[][];
    branchData: number[][];
  }) {
    const response = await fetch(`${this.baseURL}/api/flow-compute/calculate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    return response.json();
  }
  
  async getTemplate() {
    const response = await fetch(`${this.baseURL}/api/flow-compute/template`);
    return response.json();
  }
}

// 使用示例
const api = new PowerFlowAPI();

// 获取模板数据
const template = await api.getTemplate();

// 执行计算
const result = await api.calculatePowerFlow({
  baseVoltage: 10.3,
  basePower: 10.38,
  convergencePrecision: 1e-6,
  busData: template.data.busData,
  branchData: template.data.branchData,
});

// PSO 无功优化示例
const psoResult = await api.psoOptimize({
  busData: template.data.busData,
  branchData: template.data.branchData,
  tunableNodes: [
    [32, -0.3, 0.3, "节点33"],
    [33, -0.5, 0.5, "节点34"]
  ],
  psoParameters: {
    num_particles: 30,
    max_iter: 50,
    w: 0.8,
    c1: 1.5,
    c2: 1.5,
    v_min: 0.95,
    v_max: 1.05
  }
});
```

### JavaScript 集成

```javascript
// 潮流计算函数
async function calculatePowerFlow(busData, branchData, parameters) {
  try {
    const response = await fetch('http://localhost:5000/api/flow-compute/calculate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        busData: busData,
        branchData: branchData,
        baseVoltage: parameters.baseVoltage,
        basePower: parameters.basePower,
        convergencePrecision: parameters.convergencePrecision
      }),
    });
    
    const result = await response.json();
    
    if (result.status === 'success') {
      console.log('计算成功:', result.summary);
      return result;
    } else {
      console.error('计算失败:', result.message);
      return null;
    }
  } catch (error) {
    console.error('请求失败:', error);
    return null;
  }
}

// PSO 无功优化函数
async function psoOptimize(busData, branchData, tunableNodes, psoParameters) {
  try {
    const response = await fetch('http://localhost:5000/api/pso-optimize/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        busData: busData,
        branchData: branchData,
        tunableNodes: tunableNodes,
        psoParameters: psoParameters
      }),
    });
    
    const result = await response.json();
    
    if (result.status === 'success') {
      console.log('PSO优化成功:', result.data);
      console.log(`网损率降低: ${result.data.loss_reduction_percent}%`);
      return result;
    } else {
      console.error('PSO优化失败:', result.message);
      return null;
    }
  } catch (error) {
    console.error('请求失败:', error);
    return null;
  }
}

// 使用示例
const busData = [[1,0,0],[2,0.5,0.1],[3,-0.2,0]];
const branchData = [[1,1,2,0.1,0.2],[2,2,3,0.05,0.1]];
const tunableNodes = [[2, -0.1, 0.1, "节点2"]];

// 执行潮流计算
const flowResult = await calculatePowerFlow(busData, branchData, {
  baseVoltage: 10.3,
  basePower: 10.38,
  convergencePrecision: 1e-6
});

// 执行PSO优化
const psoResult = await psoOptimize(busData, branchData, tunableNodes, {
  num_particles: 30,
  max_iter: 50,
  w: 0.8,
  c1: 1.5,
  c2: 1.5,
  v_min: 0.95,
  v_max: 1.05
});
```

## 测试工具

### API测试客户端

```bash
# 运行API测试
python api_client.py
```

测试客户端将自动验证所有API接口功能。

### 手动测试

```bash
# 健康检查
curl http://localhost:5000/api/health

# 获取模板数据
curl http://localhost:5000/api/flow-compute/template

# 执行计算（使用模板数据）
curl -X POST http://localhost:5000/api/flow-compute/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "baseVoltage": 10.3,
    "basePower": 10.38,
    "convergencePrecision": 1e-6,
    "busData": [[1,0,0],[2,0.5,0.1],[3,-0.2,0]],
    "branchData": [[1,1,2,0.1,0.2],[2,2,3,0.05,0.1]]
  }'

# 获取PSO模板数据
curl http://localhost:5000/api/pso-optimize/template

# 执行PSO优化
curl -X POST http://localhost:5000/api/pso-optimize/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "busData": [[1,0,0],[2,0.5,0.1],[3,-0.2,0]],
    "branchData": [[1,1,2,0.1,0.2],[2,2,3,0.05,0.1]],
    "tunableNodes": [[2, -0.1, 0.1, "节点2"]],
    "psoParameters": {
      "num_particles": 30,
      "max_iter": 50,
      "w": 0.8,
      "c1": 1.5,
      "c2": 1.5,
      "v_min": 0.95,
      "v_max": 1.05
    }
  }'
```

## 部署说明

### 开发环境

```bash
# 启动开发服务器
python start_server.py --debug
```

### 生产环境

```bash
# 使用Gunicorn部署
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 或使用uWSGI
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app
```

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 更换端口启动
   python start_server.py --port 5001
   ```

2. **依赖安装失败**
   ```bash
   # 使用国内镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **跨域请求失败**
   - 确保Flask-CORS已正确安装
   - 检查前端请求地址是否正确

4. **计算不收敛**
   - 检查网络拓扑是否为辐射状
   - 调整收敛精度参数
   - 验证输入数据格式

### 日志查看

```bash
# 查看服务日志
tail -f logs/app.log

# 实时监控API请求
curl -N http://localhost:5000/api/health
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues：GitHub Issues
- 邮箱：your-email@example.com