"""
TD3训练数据读取模块
从已上传的训练数据目录中读取配置和训练样本
参考 td3onesample.py 的实现
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def read_voltage_limits(line_dir: str, line_name: str) -> Tuple[float, float]:
    """
    读取电压上下限（从modeldata/volcst_{line_name}.xlsx）
    
    Args:
        line_dir: 线路目录路径
        line_name: 线路名称（如 C5336）
    
    Returns:
        (电压下限, 电压上限)
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    volcst_file = os.path.join(line_dir, 'modeldata', f'volcst_{line_name}.xlsx')
    if not os.path.exists(volcst_file):
        raise FileNotFoundError(f'电压限值文件不存在: {volcst_file}')
    
    try:
        df = pd.read_excel(volcst_file, header=None)
        # 第二行第一列=下限，第二行第二列=上限
        voltage_lower = float(df.iloc[1, 0])
        voltage_upper = float(df.iloc[1, 1])
        return voltage_lower, voltage_upper
    except Exception as e:
        raise ValueError(f'读取电压限值文件失败: {str(e)}')


def read_key_nodes(line_dir: str, line_name: str) -> List[int]:
    """
    读取关键节点索引（从modeldata/kvnd_{line_name}.xlsx）
    
    Args:
        line_dir: 线路目录路径
        line_name: 线路名称
    
    Returns:
        关键节点索引列表（节点号-1转换为索引）
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    kvnd_file = os.path.join(line_dir, 'modeldata', f'kvnd_{line_name}.xlsx')
    if not os.path.exists(kvnd_file):
        raise FileNotFoundError(f'关键节点文件不存在: {kvnd_file}')
    
    try:
        df = pd.read_excel(kvnd_file, header=0)  # 第一行是列名
        # 从第二行第一列开始读取节点号，转换为索引（-1）
        key_node_nums = df.iloc[:, 0].dropna().astype(int).tolist()
        key_node_indices = [num - 1 for num in key_node_nums]  # 节点号转索引
        return key_node_indices
    except Exception as e:
        raise ValueError(f'读取关键节点文件失败: {str(e)}')


def read_branch_data(line_dir: str, line_name: str) -> np.ndarray:
    """
    读取支路数据（从modeldata/branch_{line_name}.xlsx）
    返回二维数组（每行5列：线路号	首节点	末节点	电阻	电抗）
    
    Args:
        line_dir: 线路目录路径
        line_name: 线路名称
    
    Returns:
        二维支路数据数组 (n_branches, 5)
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    branch_file = os.path.join(line_dir, 'modeldata', f'branch_{line_name}.xlsx')
    if not os.path.exists(branch_file):
        raise FileNotFoundError(f'支路数据文件不存在: {branch_file}')
    
    try:
        df = pd.read_excel(branch_file, header=0)  # 列名：线路号	首节点	末节点	电阻	电抗
        # 按行读取，直接返回二维数组
        branch_data = []
        for _, row in df.iterrows():
            branch_data.append([
                row.iloc[0],  # 线路号
                row.iloc[1],  # 首节点
                row.iloc[2],  # 末节点
                row.iloc[3],  # 电阻
                row.iloc[4]   # 电抗
            ])
        return np.array(branch_data)
    except Exception as e:
        raise ValueError(f'读取支路数据文件失败: {str(e)}')


def read_tunable_q_nodes(line_dir: str, line_name: str, bus_data: np.ndarray) -> List[Tuple[int, float, float, str]]:
    """
    读取并计算可调节无功节点配置（从modeldata/pv_{line_name}.xlsx）
    
    Args:
        line_dir: 线路目录路径
        line_name: 线路名称
        bus_data: Bus矩阵数据（用于获取当前有功值）
    
    Returns:
        TUNABLE_Q_NODES格式的列表：[(节点索引, 无功下限, 无功上限, 节点名称), ...]
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    pv_file = os.path.join(line_dir, 'modeldata', f'pv_{line_name}.xlsx')
    if not os.path.exists(pv_file):
        raise FileNotFoundError(f'可调无功节点文件不存在: {pv_file}')
    
    try:
        df = pd.read_excel(pv_file, header=0)  # 列名：节点号 容量 调度命名
        tunable_q_nodes = []
        
        for _, row in df.iterrows():
            # 读取基础信息
            node_num = int(row.iloc[0])  # 光伏节点号
            capacity = float(row.iloc[1])  # 容量
            node_name = row.iloc[2] if not pd.isna(row.iloc[2]) else f"节点{node_num}"
            
            # 计算可调无功上下限
            node_index = node_num - 1  # 转索引
            if node_index >= bus_data.shape[0]:
                raise ValueError(f'节点号 {node_num} 超出Bus数据范围')
            
            p_current = bus_data[node_index, 1]  # Bus矩阵中该节点的有功值
            q_max = np.sqrt(max(0, capacity**2 - p_current**2))  # 无功上限
            q_min = -q_max  # 无功下限
            
            # 添加到列表
            tunable_q_nodes.append((node_index, q_min, q_max, node_name))
        
        return tunable_q_nodes
    except Exception as e:
        raise ValueError(f'读取可调无功节点文件失败: {str(e)}')


def read_training_sample(line_dir: str, line_name: str) -> Tuple[np.ndarray, float]:
    """
    读取训练样本数据（从train目录下的第一个Excel文件）
    
    Args:
        line_dir: 线路目录路径
        line_name: 线路名称
    
    Returns:
        (Bus矩阵数据, 基准电压UB)
    
    Raises:
        FileNotFoundError: 训练样本文件不存在
        ValueError: 文件格式错误
    """
    train_dir = os.path.join(line_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'训练样本目录不存在: {train_dir}')
    
    # 获取所有训练样本文件
    sample_files = glob.glob(os.path.join(train_dir, f'{line_name}_*.xlsx'))
    if not sample_files:
        raise FileNotFoundError(f'未找到训练样本文件: {train_dir}/{line_name}_*.xlsx')
    
    # 使用第一个文件
    file_path = sample_files[0]
    
    try:
        # 读取Bus数据（sheet=bus）
        df_bus = pd.read_excel(file_path, sheet_name="bus", header=0)  # 列名：节点号 有功值 无功值
        bus_data = []
        for _, row in df_bus.iterrows():
            bus_data.append([
                int(row.iloc[0]),    # 节点号
                float(row.iloc[1]),  # 有功值
                float(row.iloc[2])   # 无功值
            ])
        # 确保Bus是二维数组
        bus_array = np.array(bus_data)
        
        # 读取基准电压UB（sheet=slack）
        df_slack = pd.read_excel(file_path, sheet_name="slack")
        ub = float(df_slack.iloc[0, 0])  # slack sheet的第一个值
        
        return bus_array, ub
    except Exception as e:
        raise ValueError(f'读取训练样本文件失败: {str(e)}')


def load_training_data(line_name: str, training_data_dir: str) -> Dict:
    """
    加载指定线路的所有训练数据和配置
    
    Args:
        line_name: 线路名称（如 C5336）
        training_data_dir: 训练数据根目录
    
    Returns:
        包含所有训练数据的字典:
        {
            'bus_data': np.ndarray,
            'branch_data': np.ndarray,
            'voltage_limits': (v_min, v_max),
            'key_nodes': List[int],
            'tunable_q_nodes': List[Tuple],
            'ub': float
        }
    
    Raises:
        FileNotFoundError: 线路目录或必需文件不存在
        ValueError: 文件格式错误
    """
    # 检查线路目录是否存在
    line_dir = os.path.join(training_data_dir, line_name)
    if not os.path.exists(line_dir):
        raise FileNotFoundError(f'线路目录不存在: {line_dir}')
    
    if not os.path.isdir(line_dir):
        raise ValueError(f'路径不是目录: {line_dir}')
    
    # 读取训练样本（Bus数据和UB）
    bus_data, ub = read_training_sample(line_dir, line_name)
    
    # 读取基础配置
    voltage_lower, voltage_upper = read_voltage_limits(line_dir, line_name)
    key_nodes = read_key_nodes(line_dir, line_name)
    branch_data = read_branch_data(line_dir, line_name)
    
    # 读取并计算可调无功节点（需要bus_data）
    tunable_q_nodes = read_tunable_q_nodes(line_dir, line_name, bus_data)
    
    return {
        'bus_data': bus_data,
        'branch_data': branch_data,
        'voltage_limits': (voltage_lower, voltage_upper),
        'key_nodes': key_nodes,
        'tunable_q_nodes': tunable_q_nodes,
        'ub': ub
    }

