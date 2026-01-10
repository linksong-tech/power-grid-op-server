#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习智能体实时控制系统（无PyTorch版本）
功能：加载训练好的RL模型，接收关键节点电压（当前写死，后续对接SCADA），输出无功调节策略
适配：.npz格式模型文件，纯NumPy推理
"""
import numpy as np
import copy
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局配置（实时控制精简版） --------------------------
# 获取程序所在目录
PROJECT_ROOT = Path(__file__).parent.absolute()
# 配置文件路径（仅保留可调无功节点配置）
PV_CONFIG_PATH = PROJECT_ROOT / "POWERdata" / "C5336" / "modeldata" / "pv_C5336.xlsx"
# 模型文件路径（替换为你的.npz模型文件路径）
MODEL_PATH = "M1_0103_194228.npz"

# 全局参数（实时控制核心参数）
SB = 10  # 基准功率 MVA
UB = 10.38  # 基准电压 kV（可根据实际系统调整）
DTYPE = np.float32  # 与原Torch float32保持一致

# 电压约束（根据实际系统配置）
V_MIN = 10.0  # 电压下限 kV
V_MAX = 10.7  # 电压上限 kV

# -------------------------- 自动检测模型维度（新增：解决维度不匹配） --------------------------
def get_model_state_dim(model_path):
    """读取模型文件，自动获取训练时的状态维度"""
    try:
        model_data = np.load(model_path)
        w1 = model_data["actor_w1"]
        state_dim = w1.shape[0]  # w1的第一个维度就是状态维度
        model_data.close()
        print(f"✅ 自动检测模型训练时的状态维度：{state_dim}")
        return state_dim
    except Exception as e:
        raise ValueError(f"检测模型维度失败：{e}")

# 自动获取关键节点数量（替换硬编码）
MODEL_STATE_DIM = get_model_state_dim(MODEL_PATH)
# 关键节点索引（改为与模型维度匹配，后续对接SCADA时需传对应数量的电压）
KEY_NODES = list(range(MODEL_STATE_DIM))  # 示例：[0,1,2,3,4,5,6,7,8]（9个节点）
# 写死的SCADA电压（长度匹配模型维度）
SCADA_KEY_NODE_VOLTAGES = [10.25, 10.30, 10.35, 10.28, 10.32, 10.29, 10.31, 10.27, 10.33]  # 9个电压值

# -------------------------- NumPy版本Actor网络（核心推理模块） --------------------------
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数（对齐Torch的nn.ReLU）"""
    return np.maximum(0, x).astype(DTYPE)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数（对齐Torch的nn.Tanh）"""
    return np.tanh(x).astype(DTYPE)

class ActorNetworkNumpy:
    """纯NumPy实现的Actor网络（仅保留推理功能）"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        # 初始化参数容器
        self.params = {
            "w1": None, "b1": None,
            "w2": None, "b2": None,
            "w3": None, "b3": None,
            "w4": None, "b4": None
        }
    
    def load_params(self, params_dict: dict):
        """加载从npz文件读取的Actor参数"""
        for key in self.params.keys():
            npz_key = f"actor_{key}"
            if npz_key not in params_dict:
                raise ValueError(f"模型参数缺少{npz_key}，可用参数：{list(params_dict.keys())}")
            self.params[key] = params_dict[npz_key].astype(DTYPE)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播（仅推理）"""
        x = relu(np.dot(state, self.params["w1"]) + self.params["b1"])
        x = relu(np.dot(x, self.params["w2"]) + self.params["b2"])
        x = relu(np.dot(x, self.params["w3"]) + self.params["b3"])
        x = tanh(np.dot(x, self.params["w4"]) + self.params["b4"])
        return self.max_action * x

# -------------------------- 实时推理类（核心控制逻辑） --------------------------
class RLRealTimeController:
    """强化学习实时控制器"""
    def __init__(self, model_path, key_nodes, v_min, v_max):
        self.key_nodes = key_nodes
        self.v_min = v_min
        self.v_max = v_max
        self.state_dim = len(key_nodes)
        
        # 加载可调无功节点配置
        self.tunable_nodes = self._load_pv_config()
        self.action_dim = len(self.tunable_nodes)
        self.max_action = 1.0
        
        # 初始化并加载Actor网络
        self.actor = ActorNetworkNumpy(self.state_dim, self.action_dim, self.max_action)
        self._load_model(model_path)
        
        print(f"✅ 实时控制器初始化完成")
        print(f"   - 关键节点数量：{self.state_dim}，索引：{self.key_nodes}")
        print(f"   - 可调无功节点数量：{self.action_dim}")
        print(f"   - 电压约束：{v_min}kV ~ {v_max}kV")
    
    def _load_pv_config(self):
        """读取可调无功节点配置（精简版）"""
        if not PV_CONFIG_PATH.exists():
            raise FileNotFoundError(f"光伏配置文件不存在：{PV_CONFIG_PATH}")
        
        df = pd.read_excel(PV_CONFIG_PATH)
        if df.shape[1] < 3 or df.shape[0] < 1:
            raise ValueError("pv_C5336.xlsx格式错误或无数据")
        
        tunable_nodes = []
        for idx, row in df.iterrows():
            if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]) or pd.isna(row.iloc[2]):
                continue
            try:
                node_id = int(row.iloc[0])
                capacity = float(row.iloc[1])
                node_name = str(row.iloc[2])
                node_idx = node_id - 1  # 转换为索引
                q_max = np.sqrt(max(0, capacity**2))  # 简化无功上限计算（可根据实际逻辑调整）
                q_min = -q_max
                tunable_nodes.append((node_idx, q_min, q_max, node_name))
            except Exception as e:
                print(f"警告：第{idx+1}行数据解析失败 - {e}，跳过")
        
        if not tunable_nodes:
            raise ValueError("无有效可调无功节点配置")
        return tunable_nodes
    
    def _load_model(self, model_path):
        """加载训练好的.npz模型文件"""
        try:
            model_data = np.load(model_path)
            self.actor.load_params(model_data)
            model_data.close()
            print(f"✅ 模型加载成功：{model_path}")
        except Exception as e:
            raise ValueError(f"模型加载失败：{e}")
    
    def _normalize_voltage(self, voltages):
        """电压归一化（与训练时保持一致）"""
        normalized = (voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        return np.clip(normalized, -1, 1).astype(DTYPE)
    
    def denormalize_action(self, action):
        """动作反归一化，转换为实际无功值"""
        q_mins = np.array([node[1] for node in self.tunable_nodes])
        q_maxs = np.array([node[2] for node in self.tunable_nodes])
        
        actual_actions = []
        for i in range(len(action)):
            normalized = np.clip(action[i], -1, 1)
            actual = (normalized + 1) / 2 * (q_maxs[i] - q_mins[i]) + q_mins[i]
            actual_actions.append(actual)
        return actual_actions
    
    def get_control_strategy(self, key_node_voltages):
        """
        核心控制接口：输入关键节点电压，输出无功调节策略
        :param key_node_voltages: 关键节点电压数组（kV），长度需匹配关键节点数量
        :return: 各可调节点的无功调节值（MVar）
        """
        if len(key_node_voltages) != self.state_dim:
            raise ValueError(f"输入电压数量不匹配！期望{self.state_dim}个，实际{len(key_node_voltages)}个")
        
        # 1. 电压归一化
        normalized_state = self._normalize_voltage(np.array(key_node_voltages))
        state_input = normalized_state.reshape(1, -1)
        
        # 2. Actor网络推理
        normalized_action = self.actor.forward(state_input).flatten()
        
        # 3. 动作反归一化（转换为实际无功值）
        actual_action = self.denormalize_action(normalized_action)
        
        return actual_action
    
    def print_strategy(self, q_values):
        """格式化打印无功调节策略"""
        print("\n📋 实时无功调节策略")
        print("-" * 60)
        print(f"{'节点名称':<20} {'节点索引':<10} {'无功值(MVar)':<15} {'无功上下限(MVar)':<20}")
        print("-" * 60)
        for i, (node_idx, q_min, q_max, node_name) in enumerate(self.tunable_nodes):
            q_val = q_values[i] if i < len(q_values) else "N/A"
            q_str = f"{q_val:.4f}" if isinstance(q_val, (int, float)) else q_val
            limit_str = f"[{q_min:.4f}, {q_max:.4f}]"
            print(f"{node_name:<20} {node_idx:<10} {q_str:<15} {limit_str:<20}")
        print("-" * 60)

# -------------------------- 主函数（实时控制入口） --------------------------
if __name__ == "__main__":
    # 初始化实时控制器
    try:
        controller = RLRealTimeController(
            model_path=MODEL_PATH,
            key_nodes=KEY_NODES,
            v_min=V_MIN,
            v_max=V_MAX
        )
        
        # 获取并打印实时调节策略
        print(f"\n📡 接收SCADA关键节点电压：{SCADA_KEY_NODE_VOLTAGES} kV")
        q_strategy = controller.get_control_strategy(SCADA_KEY_NODE_VOLTAGES)
        controller.print_strategy(q_strategy)
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误：{e}")
    except ValueError as e:
        print(f"\n❌ 数据错误：{e}")
    except Exception as e:
        print(f"\n❌ 控制器异常：{e}")
        import traceback
        traceback.print_exc()