#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习智能体通用验证程序（无PyTorch版本）
功能：批量读取测试断面，自动加载配置，对比RL与PSO性能并生成评估报告+可视化图表
适配：.npz格式模型文件，纯NumPy推理，运算结果与原Torch版本严格一致
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import random
import pandas as pd
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # 忽略matplotlib字体警告

# -------------------------- 全局配置：路径定义 --------------------------
# 获取程序所在目录
PROJECT_ROOT = Path(__file__).parent.absolute()
# 配置文件路径
PV_CONFIG_PATH = PROJECT_ROOT / "POWERdata" / "C5336" / "modeldata" / "pv_C5336.xlsx"
VOLT_CONFIG_PATH = PROJECT_ROOT / "POWERdata" / "C5336" / "modeldata" / "volcst_C5336.xlsx"
KEYNODE_CONFIG_PATH = PROJECT_ROOT / "POWERdata" / "C5336" / "modeldata" / "kvnd_C5336.xlsx"
BRANCH_CONFIG_PATH = PROJECT_ROOT / "POWERdata" / "C5336" / "modeldata" / "branch_C5336.xlsx"
# 测试样本目录
TEST_DATA_DIR = PROJECT_ROOT / "POWERdata" / "C5336" / "hisdata" / "pvdatax"
# 可视化结果保存目录（保留但不再使用）
VISUALIZATION_DIR = PROJECT_ROOT / "visualization_results"
VISUALIZATION_DIR.mkdir(exist_ok=True)  # 自动创建目录

# -------------------------- 全局参数初始化 --------------------------
SB = 10  # 基准功率 MVA
UB = None  # 基准电压 kV（从测试样本读取）
pr = 1e-6  # 潮流收敛精度

# 全局配置变量（从Excel读取）
tunable_q_nodes = []  # 可调无功节点配置
v_min = None  # 电压下限
v_max = None  # 电压上限
key_nodes = []  # 关键节点索引
Branch = None  # 支路数据

# 性能评估阈值（可根据实际需求调整）
PERFORMANCE_THRESHOLDS = {
    "优秀": {"voltage_error": 0.5, "loss_error": 3.0},  # 电压误差<0.5%，网损误差<1.0%
    "良好": {"voltage_error": 1.0, "loss_error": 4.0},  # 电压误差<1.0%，网损误差<2.0%
    "合格": {"voltage_error": 2.0, "loss_error": 5.0},  # 电压误差<2.0%，网损误差<3.0%
    "不合格": {"voltage_error": float('inf'), "loss_error": float('inf')}
}

# -------------------------- NumPy版本Actor网络（严格对齐原Torch版本） --------------------------
DTYPE = np.float32  # 与原Torch float32保持一致

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数（对齐Torch的nn.ReLU）"""
    return np.maximum(0, x).astype(DTYPE)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数（对齐Torch的nn.Tanh）"""
    return np.tanh(x).astype(DTYPE)

class ActorNetworkNumpy:
    """纯NumPy实现的Actor网络（严格复刻原Torch版本结构）"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        # 初始化参数容器（与训练时的命名一致）
        self.params = {
            "w1": None, "b1": None,
            "w2": None, "b2": None,
            "w3": None, "b3": None,
            "w4": None, "b4": None
        }
    
    def load_params(self, params_dict: dict):
        """加载从npz文件读取的Actor参数"""
        # 映射npz中的参数名（actor_w1 -> w1）
        for key in self.params.keys():
            npz_key = f"actor_{key}"
            if npz_key not in params_dict:
                raise ValueError(f"模型参数缺少{npz_key}，可用参数：{list(params_dict.keys())}")
            self.params[key] = params_dict[npz_key].astype(DTYPE)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播（严格复刻原Torch的Sequential逻辑）"""
        # 输入：(1, state_dim) 输出：(1, action_dim)
        x = relu(np.dot(state, self.params["w1"]) + self.params["b1"])
        x = relu(np.dot(x, self.params["w2"]) + self.params["b2"])
        x = relu(np.dot(x, self.params["w3"]) + self.params["b3"])
        x = tanh(np.dot(x, self.params["w4"]) + self.params["b4"])
        return self.max_action * x

# -------------------------- 配置读取函数（保持不变，增加容错+格式校验） --------------------------
def load_pv_config(bus_data=None):
    """
    读取可调无功节点配置
    :param bus_data: 当前bus数据（用于计算无功上下限）
    :return: tunable_q_nodes列表
    """
    if not PV_CONFIG_PATH.exists():
        raise FileNotFoundError(f"\n光伏配置文件不存在：{PV_CONFIG_PATH}\n请检查文件路径是否正确！")
    
    df = pd.read_excel(PV_CONFIG_PATH)
    # 校验列数
    if df.shape[1] < 3:
        raise ValueError(f"\npv_C5336.xlsx格式错误：至少需要3列（节点号、容量、调度命名），当前只有{df.shape[1]}列")
    # 校验行数（至少1行数据）
    if df.shape[0] < 1:
        raise ValueError(f"\npv_C5336.xlsx无数据：文件只有表头，没有实际数据行！")
    
    tunable_nodes = []
    valid_rows = 0
    
    for idx, row in df.iterrows():
        # 跳过空行/无效行
        if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]) or pd.isna(row.iloc[2]):
            continue
        
        try:
            node_id = int(row.iloc[0])  # 光伏节点号（原始）
            capacity = float(row.iloc[1])  # 节点容量
            node_name = str(row.iloc[2])  # 调度命名
            
            # 计算节点索引（节点号-1）
            node_idx = node_id - 1
            
            # 计算无功可调上下限
            q_max = 0.0
            if bus_data is not None and node_idx < len(bus_data):
                p_current = abs(float(bus_data[node_idx, 1]))  # 当前有功值
                q_max = np.sqrt(max(0, capacity**2 - p_current**2))  # 无功上限
            
            q_min = -q_max  # 无功下限
            tunable_nodes.append((node_idx, q_min, q_max, node_name))
            valid_rows += 1
        except Exception as e:
            print(f"警告：第{idx+1}行数据解析失败 - {e}，跳过该行")
            continue
    
    if valid_rows == 0:
        raise ValueError(f"\npv_C5336.xlsx无有效数据行！请检查数据格式是否正确（节点号为整数、容量为数值）")
    
    return tunable_nodes

def load_voltage_config():
    """读取电压约束配置（增加容错）"""
    if not VOLT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"\n电压配置文件不存在：{VOLT_CONFIG_PATH}\n请检查文件路径是否正确！")
    
    df = pd.read_excel(VOLT_CONFIG_PATH)
    # 校验行数（至少1行数据，支持两种格式：表头+数据行 / 直接数据行）
    if df.shape[0] == 0:
        raise ValueError(f"\nvolcst_C5336.xlsx为空文件！")
    
    # 尝试读取数据（兼容两种格式）
    v_min_val, v_max_val = None, None
    if df.shape[0] >= 2:
        # 格式1：第一行表头，第二行数据
        try:
            v_min_val = float(df.iloc[1, 0])  # 第二行第一列：电压下限
            v_max_val = float(df.iloc[1, 1])  # 第二行第二列：电压上限
        except IndexError:
            # 第二行列数不足，尝试第一行（无表头）
            v_min_val = float(df.iloc[0, 0])
            v_max_val = float(df.iloc[0, 1])
    else:
        # 格式2：无表头，直接第一行数据
        try:
            v_min_val = float(df.iloc[0, 0])  # 第一行第一列：电压下限
            v_max_val = float(df.iloc[0, 1])  # 第一行第二列：电压上限
        except IndexError:
            raise ValueError(f"\nvolcst_C5336.xlsx格式错误：\n要求至少包含两列数据（电压下限、电压上限），当前只有{df.shape[1]}列！")
    
    # 校验数值有效性
    if v_min_val >= v_max_val:
        raise ValueError(f"\n电压约束值错误：下限({v_min_val}kV) >= 上限({v_max_val}kV)！")
    
    print(f"成功读取电压约束：下限={v_min_val}kV，上限={v_max_val}kV")
    return v_min_val, v_max_val

def load_keynode_config():
    """读取关键节点配置（增加容错）"""
    if not KEYNODE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"\n关键节点配置文件不存在：{KEYNODE_CONFIG_PATH}\n请检查文件路径是否正确！")
    
    df = pd.read_excel(KEYNODE_CONFIG_PATH)
    # 校验行数
    if df.shape[0] < 1:
        raise ValueError(f"\nkvnd_C5336.xlsx无数据：文件为空！")
    
    keynodes = []
    valid_rows = 0
    
    for idx, row in df.iterrows():
        # 跳过表头/空行（如果第一列是文本表头，自动跳过）
        if pd.isna(row.iloc[0]):
            continue
        try:
            # 尝试转换为整数（节点号）
            node_id = int(row.iloc[0])
            node_idx = node_id - 1  # 转换为索引
            keynodes.append(node_idx)
            valid_rows += 1
        except (ValueError, IndexError):
            # 跳过非数值行（表头）
            continue
    
    if valid_rows == 0:
        raise ValueError(f"\nkvnd_C5336.xlsx无有效关键节点数据！请检查数据格式（节点号为整数）")
    
    print(f"成功读取关键节点：共{len(keynodes)}个，索引={keynodes}")
    return keynodes

def load_branch_config():
    """读取支路数据配置（增加容错）"""
    if not BRANCH_CONFIG_PATH.exists():
        raise FileNotFoundError(f"\n支路配置文件不存在：{BRANCH_CONFIG_PATH}\n请检查文件路径是否正确！")
    
    df = pd.read_excel(BRANCH_CONFIG_PATH)
    # 校验列数（至少5列：线路号、首节点、末节点、电阻、电抗）
    if df.shape[1] < 5:
        raise ValueError(f"\nbranch_C5336.xlsx格式错误：至少需要5列数据，当前只有{df.shape[1]}列！")
    # 校验行数
    if df.shape[0] < 1:
        raise ValueError(f"\nbranch_C5336.xlsx无数据：文件只有表头，没有实际数据行！")
    
    branch_data = []
    valid_rows = 0
    
    for idx, row in df.iterrows():
        # 跳过空行/表头行
        if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]) or pd.isna(row.iloc[2]) or pd.isna(row.iloc[3]) or pd.isna(row.iloc[4]):
            continue
        
        try:
            line_id = int(row.iloc[0])
            start_node = int(row.iloc[1])
            end_node = int(row.iloc[2])
            r = float(row.iloc[3])
            x = float(row.iloc[4])
            branch_data.append([line_id, start_node, end_node, r, x])
            valid_rows += 1
        except Exception as e:
            print(f"警告：第{idx+1}行支路数据解析失败 - {e}，跳过该行")
            continue
    
    if valid_rows == 0:
        raise ValueError(f"\nbranch_C5336.xlsx无有效支路数据！请检查数据格式（线路号/节点为整数，电阻/电抗为数值）")
    
    print(f"成功读取支路数据：共{valid_rows}条")
    return np.array(branch_data)

def load_test_samples():
    """加载所有测试样本（增加容错）"""
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"\n测试样本目录不存在：{TEST_DATA_DIR}\n请检查目录路径是否正确！")
    
    # 匹配所有C5336_*.xlsx文件
    test_files = glob.glob(str(TEST_DATA_DIR / "C5336_*.xlsx"))
    if not test_files:
        raise FileNotFoundError(f"\n测试样本目录下无有效文件：{TEST_DATA_DIR}\n请检查文件命名格式（C5336_YYYYMMDDHHMM.xlsx）")
    
    samples = []
    for file_path in test_files:
        try:
            # 检查文件是否可读取
            xl_file = pd.ExcelFile(file_path)
            required_sheets = ["date", "slack", "bus"]
            missing_sheets = [s for s in required_sheets if s not in xl_file.sheet_names]
            if missing_sheets:
                print(f"警告：{file_path} 缺少sheet：{missing_sheets}，跳过该文件")
                continue
            
            # 读取样本数据
            df_date = pd.read_excel(file_path, sheet_name="date")
            df_slack = pd.read_excel(file_path, sheet_name="slack")
            df_bus = pd.read_excel(file_path, sheet_name="bus")
            
            # 提取时间（兼容表头/无表头格式）
            sample_time = None
            if df_date.shape[0] >= 1:
                if df_date.shape[0] >= 2:
                    sample_time = str(df_date.iloc[1, 0])  # 第二行第一列
                else:
                    sample_time = str(df_date.iloc[0, 0])  # 第一行第一列
            if not sample_time or sample_time == "nan":
                # 从文件名提取时间
                file_name = Path(file_path).stem
                sample_time = file_name.replace("C5336_", "")
            
            # 提取基准电压
            slack_voltage = None
            if df_slack.shape[0] >= 1:
                if df_slack.shape[0] >= 2:
                    slack_voltage = float(df_slack.iloc[1, 0])  # 第二行第一列
                else:
                    slack_voltage = float(df_slack.iloc[0, 0])  # 第一行第一列
            if slack_voltage is None or slack_voltage <= 0:
                raise ValueError("基准电压无效（<=0）")
            
            # 提取bus数据
            bus_data = []
            valid_bus_rows = 0
            for idx, row in df_bus.iterrows():
                if idx == 0 and isinstance(row.iloc[0], str):  # 跳过表头行
                    continue
                if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]) or pd.isna(row.iloc[2]):
                    continue
                
                try:
                    node_id = int(row.iloc[0])
                    p = float(row.iloc[1])
                    q = float(row.iloc[2])
                    bus_data.extend([node_id, p, q])
                    valid_bus_rows += 1
                except Exception as e:
                    print(f"警告：{file_path} bus表第{idx+1}行解析失败 - {e}，跳过该行")
                    continue
            
            if valid_bus_rows == 0:
                print(f"警告：{file_path} bus表无有效数据，跳过该文件")
                continue
            
            bus_array = np.array(bus_data).reshape(-1, 3)  # 修正：直接reshape为n×3
            
            samples.append({
                "file_path": file_path,
                "time": sample_time,
                "ub": slack_voltage,
                "bus": bus_array  # 修正：存储reshape后的二维数组
            })
            print(f"成功加载测试样本：{sample_time}（{Path(file_path).name}）")
        except Exception as e:
            print(f"读取测试样本失败 {file_path}：{e}")
            continue
    
    if len(samples) == 0:
        raise ValueError("\n无有效测试样本！请检查样本文件格式")
    
    return samples

# -------------------------- 核心工具函数（保持不变） --------------------------
def power_flow(Bus, tunable_q_values, tunable_nodes, branch_data, sb=10, ub=10.38):
    """
    潮流计算（适配动态配置）
    :param Bus: 节点数据（n×3）
    :param tunable_q_values: 可调无功值
    :param tunable_nodes: 可调无功节点配置
    :param branch_data: 支路数据
    :param sb: 基准功率
    :param ub: 基准电压
    :return: 网损率、节点电压（kV）、功率信息
    """
    Bus_copy = copy.deepcopy(Bus)
    Branch_copy = copy.deepcopy(branch_data)
    
    # 只修改可调无功节点的Q值，其他节点保持不变
    for i, (node_idx, _, _, _) in enumerate(tunable_nodes):
        if i < len(tunable_q_values):
            Bus_copy[node_idx, 2] = tunable_q_values[i]
    
    # 功率和阻抗标幺化
    Bus_copy[:, 1] = Bus_copy[:, 1] / sb
    Bus_copy[:, 2] = Bus_copy[:, 2] / sb
    Branch_copy[:, 3] = Branch_copy[:, 3] * sb / (ub **2)
    Branch_copy[:, 4] = Branch_copy[:, 4] * sb / (ub** 2)
    
    busnum = Bus_copy.shape[0]
    branchnum = Branch_copy.shape[0]
    
    # 节点类型判断
    node_types = []
    for i in range(busnum):
        node_id = Bus_copy[i, 0]
        p = Bus_copy[i, 1]
        if node_id == 1:
            node_types.append("平衡节点")
        elif p < 0:
            node_types.append("光伏节点")
        elif p > 0:
            node_types.append("负荷节点")
        else:
            node_types.append("普通节点")
    
    # 初始化电压和相角（平衡节点电压固定为1.0标幺值）
    Vbus = np.ones(busnum)
    Vbus[0] = 1.0
    cita = np.zeros(busnum)
    
    k = 0
    Ploss = np.zeros(branchnum)
    Qloss = np.zeros(branchnum)
    P = np.zeros(branchnum)
    Q = np.zeros(branchnum)
    F = 0  # 收敛标志
    
    # 支路排序（从叶节点到根节点）
    TempBranch = Branch_copy.copy()
    s1 = np.zeros((0, 5))
    while TempBranch.size > 0:
        s = TempBranch.shape[0] - 1
        s2 = np.zeros((0, 5))
        while s >= 0:
            i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
            if i.size == 0:
                s1 = np.vstack([s1, TempBranch[s, :]]) if s1.size else TempBranch[s, :].reshape(1, -1)
            else:
                s2 = np.vstack([s2, TempBranch[s, :]]) if s2.size else TempBranch[s, :].reshape(1, -1)
            s -= 1
        TempBranch = s2.copy()
    
    # 前推回代潮流计算
    while k < 100 and F == 0:
        Pij1 = np.zeros(busnum)
        Qij1 = np.zeros(busnum)
        
        # 前推计算功率损耗和支路功率
        for s in range(branchnum):
            ii = int(s1[s, 1] - 1)
            jj = int(s1[s, 2] - 1)
            Pload = Bus_copy[jj, 1]
            Qload = Bus_copy[jj, 2]
            R = s1[s, 3]
            X = s1[s, 4]
            VV = Vbus[jj]
            
            Pij0 = Pij1[jj]
            Qij0 = Qij1[jj]
            
            II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
            Ploss[int(s1[s, 0]) - 1] = II * R
            Qloss[int(s1[s, 0]) - 1] = II * X
            
            P[int(s1[s, 0]) - 1] = Pload + Ploss[int(s1[s, 0]) - 1] + Pij0
            Q[int(s1[s, 0]) - 1] = Qload + Qloss[int(s1[s, 0]) - 1] + Qij0
            
            Pij1[ii] += P[int(s1[s, 0]) - 1]
            Qij1[ii] += Q[int(s1[s, 0]) - 1]
        
        # 回代计算节点电压
        for s in range(branchnum-1, -1, -1):
            ii = int(s1[s, 2] - 1)
            kk = int(s1[s, 1] - 1)
            R = s1[s, 3]
            X = s1[s, 4]
            
            V_real = Vbus[kk] - (P[int(s1[s, 0]) - 1]*R + Q[int(s1[s, 0]) - 1]*X) / Vbus[kk]
            V_imag = (P[int(s1[s, 0]) - 1]*X - Q[int(s1[s, 0]) - 1]*R) / Vbus[kk]
            
            Vbus[ii] = np.sqrt(V_real**2 + V_imag**2)
            cita[ii] = cita[kk] - np.arctan2(V_imag, V_real)
        
        # 校验收敛
        Pij2 = np.zeros(busnum)
        Qij2 = np.zeros(busnum)
        for s in range(branchnum):
            ii = int(s1[s, 1] - 1)
            jj = int(s1[s, 2] - 1)
            Pload = Bus_copy[jj, 1]
            Qload = Bus_copy[jj, 2]
            R = s1[s, 3]
            X = s1[s, 4]
            VV = Vbus[jj]
            
            Pij0 = Pij2[jj]
            Qij0 = Qij2[jj]
            
            II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
            P_val = Pload + II * R + Pij0
            Q_val = Qload + II * X + Qij0
            
            Pij2[ii] += P_val
            Qij2[ii] += Q_val
        
        ddp = np.max(np.abs(Pij1 - Pij2))
        ddq = np.max(np.abs(Qij1 - Qij2))
        if ddp < pr and ddq < pr:
            F = 1
        k += 1
    
    if k == 100:
        print("潮流计算未收敛！")
        return None, None, None
    
    # 计算网损率
    P1 = np.sum(Ploss)
    balance_node_output = Pij2[0] * sb
    pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
    pv_total_injection = sum(-Bus_copy[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * sb
    total_input_power = balance_node_output + pv_total_injection
    
    load_nodes_mask = [typ == "负荷节点" for typ in node_types]
    total_output_power = sum(Bus_copy[i, 1] for i in range(busnum) if load_nodes_mask[i]) * sb
    
    loss_rate = (total_input_power - total_output_power) / total_input_power * 100 if total_input_power != 0 else 0.0
    Vbus_kv = Vbus * ub  # 转换为实际kV电压
    
    return loss_rate, Vbus_kv, (balance_node_output, pv_total_injection, total_input_power, total_output_power)

def get_observed_voltages(Bus, tunable_nodes, branch_data, sb=10, ub=10.38):
    """获取观测电压（通过潮流计算模拟传感器数据）"""
    initial_q = [Bus[node[0], 2] for node in tunable_nodes]  # 原始Q值
    _, observed_voltages, _ = power_flow(Bus, initial_q, tunable_nodes, branch_data, sb, ub)
    return observed_voltages

def pso_optimization(Bus, tunable_nodes, branch_data, v_min, v_max, sb=10, ub=10.38):
    """PSO优化（用于对比）"""
    num_particles = 20
    max_iter = 50
    w = 0.8
    c1 = 1.5
    c2 = 1.5
    
    dim = len(tunable_nodes)
    q_mins = np.array([node[1] for node in tunable_nodes])
    q_maxs = np.array([node[2] for node in tunable_nodes])
    
    particles = np.random.rand(num_particles, dim)
    for i in range(dim):
        particles[:, i] = particles[:, i] * (q_maxs[i] - q_mins[i]) + q_mins[i]
    
    velocities = np.zeros((num_particles, dim))
    pbest = np.copy(particles)
    pbest_fitness = np.ones(num_particles) * np.inf
    
    for i in range(num_particles):
        loss_rate, voltages, _ = power_flow(Bus, particles[i], tunable_nodes, branch_data, sb, ub)
        if loss_rate is None:
            pbest_fitness[i] = np.inf
            continue
        voltage_violation = np.sum(np.maximum(v_min - voltages, 0) + np.maximum(voltages - v_max, 0))
        pbest_fitness[i] = loss_rate + 100 * voltage_violation if voltage_violation > 0 else loss_rate
    
    gbest_idx = np.argmin(pbest_fitness)
    gbest = np.copy(pbest[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    
    for iter in range(max_iter):
        current_w = w - (w - 0.4) * (iter / max_iter)
        
        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = current_w * velocities[i] + \
                           c1 * r1 * (pbest[i] - particles[i]) + \
                           c2 * r2 * (gbest - particles[i])
            
            max_vel = 0.1 * (q_maxs - q_mins)
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            particles[i] += velocities[i]
            for j in range(dim):
                particles[i, j] = np.clip(particles[i, j], q_mins[j], q_maxs[j])
            
            loss_rate, voltages, _ = power_flow(Bus, particles[i], tunable_nodes, branch_data, sb, ub)
            if loss_rate is None:
                current_fitness = np.inf
            else:
                voltage_violation = np.sum(np.maximum(v_min - voltages, 0) + np.maximum(voltages - v_max, 0))
                current_fitness = loss_rate + 100 * voltage_violation if voltage_violation > 0 else loss_rate
            
            if current_fitness < pbest_fitness[i]:
                pbest[i] = np.copy(particles[i])
                pbest_fitness[i] = current_fitness
        
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest = np.copy(pbest[current_best_idx])
            gbest_fitness = pbest_fitness[current_best_idx]
    
    return gbest, gbest_fitness

# -------------------------- NumPy版本RL推理类（替换原Torch版本） --------------------------
class TD3InferenceNumpy:
    """纯NumPy实现的TD3推理类（严格对齐原Torch版本逻辑）"""
    def __init__(self, state_dim, action_dim, max_action, model_path, v_min, v_max, key_nodes):
        print(f"\n使用设备: CPU (NumPy float32)")
        
        # 初始化NumPy版本Actor网络
        self.actor = ActorNetworkNumpy(state_dim, action_dim, max_action)
        
        # 加载.npz模型文件
        try:
            model_data = np.load(model_path)
            self.actor.load_params(model_data)
            model_data.close()
            print(f"成功加载模型：{model_path}")
        except Exception as e:
            raise ValueError(f"\n模型加载失败：{e}\n请检查模型文件路径和完整性！")
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 全局电压约束
        self.v_min = v_min
        self.v_max = v_max
        
        # 全局关键节点
        self.key_nodes = key_nodes
    
    def _build_state(self, observed_voltages):
        """构建状态（从观测电压中提取关键节点电压）"""
        # 1. 提取关键节点的观测电压
        key_node_voltages = observed_voltages[self.key_nodes]  # 形状：(n,)
        
        # 2. 电压归一化（与训练时完全一致）
        normalized_voltages = (key_node_voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        normalized_voltages = np.clip(normalized_voltages, -1, 1)
        
        return normalized_voltages.astype(DTYPE)
    
    def denormalize_action(self, action, tunable_nodes):
        """动作反归一化（与训练时完全一致）"""
        q_mins = np.array([node[1] for node in tunable_nodes])
        q_maxs = np.array([node[2] for node in tunable_nodes])
        
        actual_actions = []
        for i in range(len(action)):
            normalized = np.clip(action[i], -1, 1)
            actual = (normalized + 1) / 2 * (q_maxs[i] - q_mins[i]) + q_mins[i]
            actual_actions.append(actual)
        return actual_actions
    
    def predict(self, observed_voltages, tunable_nodes):
        """预测最优无功配置（纯NumPy推理）"""
        # 构建状态
        state = self._build_state(observed_voltages)
        # 转换为模型输入格式 (1, state_dim)
        state_input = state.reshape(1, -1)
        
        # 模型前向传播（无梯度计算，纯推理）
        normalized_action = self.actor.forward(state_input).flatten()
        
        # 反归一化
        actual_action = self.denormalize_action(normalized_action, tunable_nodes)
        return actual_action

# -------------------------- 性能评估函数（保持不变） --------------------------
def calculate_errors(rl_voltages, pso_voltages, rl_loss, pso_loss):
    """
    计算误差
    :param rl_voltages: RL优化后的节点电压
    :param pso_voltages: PSO优化后的节点电压
    :param rl_loss: RL网损率
    :param pso_loss: PSO网损率
    :return: 电压平均误差(%)，网损误差(%)
    """
    # 电压平均误差（MAE百分比）
    voltage_errors = np.abs(rl_voltages - pso_voltages) / pso_voltages * 100
    avg_voltage_error = np.mean(voltage_errors)
    
    # 网损误差（绝对值百分比）
    loss_error = np.abs(rl_loss - pso_loss) / pso_loss * 100 if pso_loss != 0 else float('inf')
    
    return avg_voltage_error, loss_error

def evaluate_performance(voltage_error, loss_error):
    """性能分级评估"""
    if voltage_error <= PERFORMANCE_THRESHOLDS["优秀"]["voltage_error"] and \
       loss_error <= PERFORMANCE_THRESHOLDS["优秀"]["loss_error"]:
        return "优秀"
    elif voltage_error <= PERFORMANCE_THRESHOLDS["良好"]["voltage_error"] and \
         loss_error <= PERFORMANCE_THRESHOLDS["良好"]["loss_error"]:
        return "良好"
    elif voltage_error <= PERFORMANCE_THRESHOLDS["合格"]["voltage_error"] and \
         loss_error <= PERFORMANCE_THRESHOLDS["合格"]["loss_error"]:
        return "合格"
    else:
        return "不合格"

# -------------------------- 辅助打印函数（保持不变） --------------------------
def print_reactive_power_strategy(tunable_nodes, rl_q, pso_q):
    """
    打印RL和PSO的无功调节策略（关联节点名称）
    :param tunable_nodes: 可调无功节点配置列表
    :param rl_q: RL输出的无功值列表
    :param pso_q: PSO输出的无功值列表
    """
    print(f"\n⚙️  无功调节策略对比：")
    print(f"{'节点名称':<20} {'RL无功值(MVar)':<18} {'PSO无功值(MVar)':<18} {'无功上下限(MVar)':<20}")
    print("-" * 76)
    for i, (node_idx, q_min, q_max, node_name) in enumerate(tunable_nodes):
        rl_val = rl_q[i] if i < len(rl_q) else "N/A"
        pso_val = pso_q[i] if i < len(pso_q) else "N/A"
        # 格式化输出，保留4位小数
        rl_str = f"{rl_val:.4f}" if isinstance(rl_val, (int, float)) else rl_val
        pso_str = f"{pso_val:.4f}" if isinstance(pso_val, (int, float)) else pso_val
        limit_str = f"[{q_min:.4f}, {q_max:.4f}]"
        print(f"{node_name:<20} {rl_str:<18} {pso_str:<18} {limit_str:<20}")

# -------------------------- 可视化函数（保持不变，仅保留指定3个图表） --------------------------
def setup_plot_style():
    """设置绘图样式（修复matplotlib版本兼容和字体问题）"""
    # 适配macOS的中文字体
    plt.rcParams["font.family"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["savefig.dpi"] = 300

def plot_error_distribution(df_results):
    """
    图3：电压误差和网损误差分布箱线图（改为直接显示）
    """
    setup_plot_style()
    
    # 准备数据
    voltage_errors = df_results["电压平均误差(%)"].values
    loss_errors = df_results["网损误差(%)"].values
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # 电压误差箱线图
    bp1 = ax1.boxplot(voltage_errors, patch_artist=True, labels=['电压平均误差'])
    bp1['boxes'][0].set_facecolor('#85C1E9')
    ax1.set_title('电压平均误差分布', fontsize=14, fontweight='bold')
    ax1.set_ylabel('误差值 (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax1.text(0.7, 0.95, 
             f'均值: {np.mean(voltage_errors):.2f}%\n中位数: {np.median(voltage_errors):.2f}%\n最大值: {np.max(voltage_errors):.2f}%\n最小值: {np.min(voltage_errors):.2f}%',
             transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    # 网损误差箱线图
    bp2 = ax2.boxplot(loss_errors, patch_artist=True, labels=['网损误差'])
    bp2['boxes'][0].set_facecolor('#F8C471')
    ax2.set_title('网损误差分布', fontsize=14, fontweight='bold')
    ax2.set_ylabel('误差值 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax2.text(0.7, 0.95, 
             f'均值: {np.mean(loss_errors):.2f}%\n中位数: {np.median(loss_errors):.2f}%\n最大值: {np.max(loss_errors):.2f}%\n最小值: {np.min(loss_errors):.2f}%',
             transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    # 整体标题
    fig.suptitle('RL相对PSO的误差分布', fontsize=16, fontweight='bold', y=0.98)
    
    # 直接显示图表（不再保存）
    plt.tight_layout()
    plt.show()

def plot_performance_distribution(df_results):
    """
    图4：性能评估等级分布饼图（修复explode参数长度问题，改为直接显示）
    """
    setup_plot_style()
    
    # 统计各等级数量
    performance_counts = df_results["性能评估"].value_counts()
    labels = performance_counts.index
    sizes = performance_counts.values
    
    # 动态生成explode数组，长度和实际等级数量一致
    explode = [0.05] * len(labels)  # 每个等级都轻微突出
    
    # 动态匹配颜色（按实际等级数量取色）
    color_map = {
        "优秀": '#27AE60',
        "良好": '#F39C12',
        "合格": '#E67E22',
        "不合格": '#E74C3C'
    }
    colors = [color_map.get(label, '#95A5A6') for label in labels]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(sizes, 
                                      explode=explode,
                                      labels=labels,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      shadow=True,
                                      startangle=90,
                                      textprops={'fontsize': 12})
    
    # 美化百分比标签
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('RL智能体性能评估等级分布', fontsize=16, fontweight='bold', pad=20)
    
    # 添加数量标注
    total = sum(sizes)
    for i, (label, size) in enumerate(zip(labels, sizes)):
        ax.text(1.3, 0.9 - i*0.15, 
                f'{label}: {size}个样本 ({size/total*100:.1f}%)',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.3))
    
    # 直接显示图表（不再保存）
    plt.tight_layout()
    plt.show()

def plot_loss_scatter(df_results):
    """
    图5：RL vs PSO 网损率散点图（相关性分析，改为直接显示）
    """
    setup_plot_style()
    
    # 准备数据
    rl_loss = df_results["RL网损率(%)"].values
    pso_loss = df_results["PSO网损率(%)"].values
    
    # 计算相关系数
    corr = np.corrcoef(rl_loss, pso_loss)[0, 1]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pso_loss, rl_loss, c='#3498DB', alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # 添加等值线（y=x）
    min_val = min(min(rl_loss), min(pso_loss))
    max_val = max(max(rl_loss), max(pso_loss))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='等值线 (RL=PSO)')
    
    # 美化
    ax.set_title(f'RL vs PSO 网损率相关性分析 (相关系数: {corr:.3f})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('PSO网损率 (%)', fontsize=12)
    ax.set_ylabel('RL网损率 (%)', fontsize=12)
    ax.legend(fontsize=11)
    
    # 添加趋势线
    z = np.polyfit(pso_loss, rl_loss, 1)
    p = np.poly1d(z)
    ax.plot(pso_loss, p(pso_loss), "g-", alpha=0.8, label=f'趋势线 (y={z[0]:.3f}x+{z[1]:.3f})')
    ax.legend(fontsize=10)
    
    # 直接显示图表（不再保存）
    plt.tight_layout()
    plt.show()

def generate_all_visualizations(df_results):
    """生成所有保留的可视化图表（改为直接显示）"""
    print(f"\n=== 开始生成可视化图表 ===")
    
    if len(df_results) == 0:
        print("⚠️  无验证结果，跳过可视化生成")
        return
    
    try:
        # 仅保留指定的3个图表
        print("📊 显示电压误差和网损误差分布箱线图...")
        plot_error_distribution(df_results)
        
        print("📊 显示RL智能体性能评估等级分布饼图...")
        plot_performance_distribution(df_results)
        
        print("📊 显示RL vs PSO 网损率相关性散点图...")
        plot_loss_scatter(df_results)
        
        print(f"\n🎉 所有可视化图表已显示完成！")
        
    except Exception as e:
        print(f"⚠️  可视化生成过程出错：{e}")
        import traceback
        traceback.print_exc()

# -------------------------- 主验证函数（适配NumPy推理） --------------------------
def batch_validate_model(model_path):
    """批量验证模型（适配NumPy版本）"""
    # 1. 加载全局配置
    global v_min, v_max, key_nodes, Branch
    print("=== 开始加载配置文件 ===")
    try:
        # 加载电压约束
        v_min, v_max = load_voltage_config()
        
        # 加载关键节点
        key_nodes = load_keynode_config()
        
        # 加载支路数据
        Branch = load_branch_config()
        
        # 2. 加载测试样本
        print("\n=== 开始加载测试样本 ===")
        test_samples = load_test_samples()
        print(f"共加载{len(test_samples)}个有效测试样本")
        
        # 3. 初始化RL推理器（NumPy版本）
        state_dim = len(key_nodes)
        # 先加载一个样本获取可调节点数量
        first_sample_bus = test_samples[0]["bus"]  # 已修正为二维数组
        action_dim = len(load_pv_config(first_sample_bus))
        max_action = 1.0
        
        print(f"\n=== 初始化RL推理器（NumPy版本）===")
        print(f"状态维度：{state_dim}（关键节点数）")
        print(f"动作维度：{action_dim}（可调无功节点数）")
        
        rl_infer = TD3InferenceNumpy(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            model_path=model_path,
            v_min=v_min,
            v_max=v_max,
            key_nodes=key_nodes
        )
        
        # 4. 批量处理样本
        results = []
        total_samples = len(test_samples)
        print(f"\n=== 开始批量验证（共{total_samples}个样本）===")
        
        for idx, sample in enumerate(test_samples):
            print(f"\n----- 处理样本 {idx+1}/{total_samples} -----")
            sample_time = sample["time"]
            ub = sample["ub"]
            Bus_reshaped = sample["bus"]  # 已修正为二维数组
            
            print(f"样本时间：{sample_time}")
            print(f"基准电压：{ub}kV")
            print(f"节点数量：{Bus_reshaped.shape[0]}")
            
            # 加载当前样本的可调无功节点配置
            try:
                tunable_q_nodes = load_pv_config(Bus_reshaped)
                print(f"可调无功节点：{[f'{node[3]}（索引{node[0]}）' for node in tunable_q_nodes]}")
            except Exception as e:
                print(f"加载可调无功节点失败：{e}，跳过该样本")
                continue
            
            # 获取观测电压
            observed_voltages = get_observed_voltages(Bus_reshaped, tunable_q_nodes, Branch, SB, ub)
            if observed_voltages is None:
                print(f"样本{sample_time}潮流计算失败，跳过")
                continue
            
            # 初始状态（优化前）
            initial_q = [Bus_reshaped[node[0], 2] for node in tunable_q_nodes]
            initial_loss, initial_voltages, _ = power_flow(Bus_reshaped, initial_q, tunable_q_nodes, Branch, SB, ub)
            # 新增：打印优化前网损率
            if initial_loss is not None:
                print(f"优化前网损率：{initial_loss:.4f}%")
            else:
                print("优化前潮流计算未收敛，跳过该样本")
                continue
            
            # PSO优化
            print("PSO优化中...")
            try:
                pso_q, _ = pso_optimization(Bus_reshaped, tunable_q_nodes, Branch, v_min, v_max, SB, ub)
                pso_loss, pso_voltages, _ = power_flow(Bus_reshaped, pso_q, tunable_q_nodes, Branch, SB, ub)
            except Exception as e:
                print(f"PSO优化失败：{e}，跳过该样本")
                continue
            
            # RL优化（NumPy版本）
            print("RL优化中...")
            try:
                rl_q = rl_infer.predict(observed_voltages, tunable_q_nodes)
                rl_loss, rl_voltages, _ = power_flow(Bus_reshaped, rl_q, tunable_q_nodes, Branch, SB, ub)
            except Exception as e:
                print(f"RL优化失败：{e}，跳过该样本")
                continue
            
            # 新增：打印无功调节策略
            print_reactive_power_strategy(tunable_q_nodes, rl_q, pso_q)
            
            # 计算误差
            try:
                # 修改：打印优化前/RL/PSO网损率对比，包含相对降幅
                print(f"\n📊 网损率对比（{sample_time}）：")
                print(f"   - 优化前网损率：{initial_loss:.4f}%")
                print(f"   - RL强化学习优化网损率：{rl_loss:.4f}%（相对优化前降低：{((initial_loss - rl_loss)/initial_loss*100):.2f}%）")
                print(f"   - PSO粒子群算法优化网损率：{pso_loss:.4f}%（相对优化前降低：{((initial_loss - pso_loss)/initial_loss*100):.2f}%）")
                
                # 计算误差
                voltage_error, loss_error = calculate_errors(rl_voltages, pso_voltages, rl_loss, pso_loss)
                
                # 打印误差结果
                print(f"🔍 误差计算结果：")
                print(f"   - 电压平均误差：{voltage_error:.4f}%")
                print(f"   - 网损误差：{loss_error:.4f}%")
                
                performance = evaluate_performance(voltage_error, loss_error)
                
                # 新增：记录优化前网损率字段
                results.append({
                    "序号": idx + 1,
                    "断面时间": sample_time,
                    "优化前网损率(%)": round(initial_loss, 4),  # 新增字段
                    "RL网损率(%)": round(rl_loss, 4),
                    "PSO网损率(%)": round(pso_loss, 4),
                    "RL相对优化前降幅(%)": round(((initial_loss - rl_loss)/initial_loss*100), 2),  # 新增字段
                    "PSO相对优化前降幅(%)": round(((initial_loss - pso_loss)/initial_loss*100), 2),  # 新增字段
                    "电压平均误差(%)": round(voltage_error, 4),
                    "网损误差(%)": round(loss_error, 4),
                    "性能评估": performance,
                    "RL无功策略": [round(val, 4) for val in rl_q],
                    "PSO无功策略": [round(val, 4) for val in pso_q]
                })
                
                print(f"✅ 处理完成 - 性能评估：{performance}")
            except Exception as e:
                print(f"计算误差失败：{e}，跳过该样本")
                continue
        
        # 5. 生成汇总报告
        print(f"\n=== 验证完成！生成汇总报告 ===")
        if len(results) == 0:
            print("警告：无有效验证结果！")
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        print("\n验证结果明细：")
        # 修改：显示优化前网损率和相对降幅字段
        print_cols = [col for col in df_results.columns if col not in ["RL无功策略", "PSO无功策略"]]
        print(df_results[print_cols].to_string(index=False))
        
        # 修改：计算并显示优化前网损率、RL/PSO相对降幅的平均值
        avg_initial_loss = np.mean([r["优化前网损率(%)"] for r in results])
        avg_rl_loss = np.mean([r["RL网损率(%)"] for r in results])
        avg_pso_loss = np.mean([r["PSO网损率(%)"] for r in results])
        avg_rl_reduction = np.mean([r["RL相对优化前降幅(%)"] for r in results])
        avg_pso_reduction = np.mean([r["PSO相对优化前降幅(%)"] for r in results])
        avg_voltage_error = np.mean([r["电压平均误差(%)"] for r in results])
        avg_loss_error = np.mean([r["网损误差(%)"] for r in results])
        
        overall_performance = evaluate_performance(avg_voltage_error, avg_loss_error)
        
        print(f"\n=== 整体性能评估 ===")
        print(f"有效样本数：{len(results)}")
        print(f"平均优化前网损率：{avg_initial_loss:.4f}%")  # 新增
        print(f"平均RL网损率：{avg_rl_loss:.4f}%（相对优化前平均降低：{avg_rl_reduction:.2f}%）")  # 修改
        print(f"平均PSO网损率：{avg_pso_loss:.4f}%（相对优化前平均降低：{avg_pso_reduction:.2f}%）")  # 修改
        print(f"平均电压误差：{avg_voltage_error:.4f}%")
        print(f"平均网损误差：{avg_loss_error:.4f}%")
        print(f"智能体整体性能：{overall_performance}")
        
        # 保存结果到Excel（包含优化前网损率和相对降幅）
        report_path = PROJECT_ROOT / "rl_verification_report.xlsx"
        try:
            df_results.to_excel(report_path, index=False)
            print(f"\n验证报告已保存至：{report_path}")
            print("📄 报告包含字段：断面时间、优化前网损率(%)、RL网损率(%)、PSO网损率(%)、RL相对优化前降幅(%)、PSO相对优化前降幅(%)、电压平均误差(%)、网损误差(%)、性能评估、RL无功策略、PSO无功策略")
        except Exception as e:
            print(f"\n保存报告失败：{e}")
        
        # 6. 生成可视化图表（改为直接显示）
        generate_all_visualizations(df_results)
        
        return df_results
    
    except Exception as e:
        print(f"\n配置加载/验证失败：{e}")
        raise

# -------------------------- 主函数（适配NumPy版本） --------------------------
if __name__ == "__main__":
    # 模型路径（替换为你的.npz模型文件路径）
    MODEL_PATH = "M1_0103_000110.npz"
    
    # 设置中文显示（适配macOS）
    plt.rcParams["font.family"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 执行批量验证
    try:
        batch_validate_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"\n【错误】文件未找到：{e}")
        print("请检查：")
        print("1. 模型文件路径是否正确（.npz格式）")
        print("2. 配置文件路径（POWERdata/C5336/modeldata/）是否存在")
        print("3. 测试样本目录（POWERdata/C5336/hisdata/pvdatax/）是否存在")
    except ValueError as e:
        print(f"\n【错误】数据格式错误：{e}")
        print("请检查配置Excel文件的格式是否符合要求")
    except Exception as e:
        print(f"\n【错误】验证过程出错：{e}")
        import traceback
        traceback.print_exc()