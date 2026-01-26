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
from pathlib import Path


def _extract_first_float(df: pd.DataFrame) -> float:
    """
    从一个 sheet 的 DataFrame 中提取第一个可解析为 float 的单元格。
    用于兼容第一行是中文标题（如“母线电压”）的情况。
    """
    max_rows = min(20, df.shape[0])
    max_cols = min(20, df.shape[1])
    for r in range(max_rows):
        for c in range(max_cols):
            val = df.iat[r, c]
            if pd.isna(val):
                continue
            try:
                f = float(val)
                if np.isfinite(f):
                    return f
            except Exception:
                continue
    raise ValueError("未找到可用的数值单元格")


def _read_bus_matrix(file_path: str) -> np.ndarray:
    """
    读取 bus sheet，兼容第一行中文标题。
    输出二维数组 (n, 3): [节点号, 有功, 无功]
    """
    df_bus = pd.read_excel(file_path, sheet_name="bus", header=None)
    bus_rows: List[List[float]] = []
    for _, row in df_bus.iterrows():
        if len(row) < 3:
            continue
        try:
            n = float(row.iloc[0])
            p = float(row.iloc[1])
            q = float(row.iloc[2])
            bus_rows.append([n, p, q])
        except Exception:
            # 跳过标题行/空行/非数值行
            continue
    if not bus_rows:
        raise ValueError("bus sheet 无有效数据行")
    return np.array(bus_rows, dtype=float)


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
        line_name: 线路名称（保留参数以兼容旧代码，实际不使用）
    
    Returns:
        (Bus矩阵数据, 基准电压UB)
    
    Raises:
        FileNotFoundError: 训练样本文件不存在
        ValueError: 文件格式错误
    """
    train_dir = os.path.join(line_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'训练样本目录不存在: {train_dir}')
    
    # 获取所有训练样本文件（不再限制文件名格式）
    sample_files = glob.glob(os.path.join(train_dir, '*.xlsx'))
    if not sample_files:
        raise FileNotFoundError(f'未找到训练样本文件: {train_dir}/*.xlsx')
    
    # 使用第一个文件
    file_path = sample_files[0]
    
    try:
        bus_array = _read_bus_matrix(file_path)

        # 读取基准电压UB（sheet=slack，兼容中文标题）
        df_slack = pd.read_excel(file_path, sheet_name="slack", header=None)
        ub = _extract_first_float(df_slack)
        
        return bus_array, ub
    except Exception as e:
        raise ValueError(f'读取训练样本文件失败: {str(e)}')

def read_test_samples(line_dir: str, line_name: str) -> List[Dict]:
    """
    读取测试样本数据（从 test 目录下的所有 Excel 文件）

    文件格式约定（与训练样本一致）：
    - sheet=bus: 节点号/有功/无功（至少3列）
    - sheet=slack: 基准电压 UB
    - sheet=date: 时间（可选，缺失则从文件名推断）

    Args:
        line_dir: 线路目录路径
        line_name: 线路名称（保留参数以兼容旧代码，实际不使用）

    Returns:
        List[Dict]: 每个样本为:
          {
            'time': str,
            'ub': float,
            'bus': List[float]  # 扁平化: [节点号, P, Q, ...]
          }
    """
    test_dir = os.path.join(line_dir, 'test')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f'测试样本目录不存在: {test_dir}')

    # 获取所有测试样本文件（不再限制文件名格式）
    sample_files = sorted(glob.glob(os.path.join(test_dir, '*.xlsx')))
    if not sample_files:
        raise FileNotFoundError(f'未找到测试样本文件: {test_dir}/*.xlsx')

    samples: List[Dict] = []
    for file_path in sample_files:
        try:
            bus_array = _read_bus_matrix(file_path)

            # UB（sheet=slack，兼容中文标题）
            df_slack = pd.read_excel(file_path, sheet_name="slack", header=None)
            ub = _extract_first_float(df_slack)

            # 时间（sheet=date，缺失则从文件名推断）
            sample_time = None
            try:
                df_date = pd.read_excel(file_path, sheet_name="date", header=None)
                # 常见格式：第0行是中文标题，第1行是值
                if df_date.shape[0] >= 2 and not pd.isna(df_date.iloc[1, 0]):
                    sample_time = str(df_date.iloc[1, 0])
                elif df_date.shape[0] >= 1 and not pd.isna(df_date.iloc[0, 0]):
                    sample_time = str(df_date.iloc[0, 0])
            except Exception:
                sample_time = None

            if not sample_time or sample_time == 'nan':
                stem = Path(file_path).stem
                # 使用文件名作为时间标识
                sample_time = stem

            samples.append({
                'time': sample_time,
                'ub': ub,
                'bus': bus_array.reshape(-1).tolist(),
            })
        except Exception as e:
            raise ValueError(f'读取测试样本文件失败: {file_path} - {str(e)}')

    return samples


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


def load_training_data_from_line_id(line_id: str, line_name: str) -> Dict:
    """
    从新的线路目录结构加载训练数据（从 line_info.json 读取配置）
    
    Args:
        line_id: 线路ID（UUID）
        line_name: 线路名称（用于匹配文件名）
    
    Returns:
        包含所有训练数据的字典
    
    Raises:
        FileNotFoundError: 线路目录或必需文件不存在
        ValueError: 文件格式错误或数据缺失
    """
    # 导入 line_service
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
    from line_service import line_service
    
    # 获取线路信息
    line_data = line_service.get_line(line_id)
    if not line_data:
        raise FileNotFoundError(f'线路不存在: {line_id}')
    
    # 检查必需的配置数据
    if not line_data.get('lineParamsData'):
        raise ValueError('线路缺少支路数据（lineParamsData），请先在线路建模中上传')
    
    if not line_data.get('voltageLimitData'):
        raise ValueError('线路缺少电压上下限数据（voltageLimitData），请先在线路建模中上传')
    
    if not line_data.get('keyNodesData'):
        raise ValueError('线路缺少关键节点数据（keyNodesData），请先在线路建模中上传')
    
    if not line_data.get('adjustablePvData'):
        raise ValueError('线路缺少可调光伏数据（adjustablePvData），请先在线路建模中上传')
    
    # 获取线路目录
    line_data_dir = line_service._get_line_dir(line_id)
    
    # 读取训练样本（Bus数据和UB）
    bus_data, ub = read_training_sample(line_data_dir, line_name)
    
    # 从 line_info.json 读取配置数据
    # 1. 支路数据 lineParamsData: [[线路号, 首节点, 末节点, 电阻, 电抗], ...]
    branch_data = np.array(line_data['lineParamsData'], dtype=float)
    
    # 2. 电压上下限 voltageLimitData: [下限, 上限]
    voltage_limit_data = line_data['voltageLimitData']
    if not isinstance(voltage_limit_data, list) or len(voltage_limit_data) != 2:
        raise ValueError('电压上下限数据格式错误，应为 [下限, 上限]')
    voltage_lower = float(voltage_limit_data[0])
    voltage_upper = float(voltage_limit_data[1])
    
    # 3. 关键节点 keyNodesData: [节点号1, 节点号2, ...]（1-based，需要转换为0-based索引）
    key_nodes_data = line_data['keyNodesData']
    key_nodes = [int(node) - 1 for node in key_nodes_data]  # 转换为0-based索引
    
    # 4. 可调光伏节点 adjustablePvData: [[节点号, 容量, 名称], ...]
    adjustable_pv_data = line_data['adjustablePvData']
    tunable_q_nodes = []
    for pv in adjustable_pv_data:
        node_num = int(pv[0])  # 节点号（1-based）
        capacity = float(pv[1])  # 容量
        node_name = str(pv[2]) if len(pv) > 2 else f"节点{node_num}"
        
        # 计算可调无功上下限
        node_index = node_num - 1  # 转换为0-based索引
        if node_index >= bus_data.shape[0]:
            raise ValueError(f'节点号 {node_num} 超出Bus数据范围')
        
        p_current = bus_data[node_index, 1]  # Bus矩阵中该节点的有功值
        q_max = np.sqrt(max(0, capacity**2 - p_current**2))  # 无功上限
        q_min = -q_max  # 无功下限
        
        tunable_q_nodes.append((node_index, q_min, q_max, node_name))
    
    return {
        'bus_data': bus_data,
        'branch_data': branch_data,
        'voltage_limits': (voltage_lower, voltage_upper),
        'key_nodes': key_nodes,
        'tunable_q_nodes': tunable_q_nodes,
        'ub': ub
    }


def load_test_samples(line_name: str, training_data_dir: str) -> List[Dict]:
    """
    加载指定线路的全部测试样本

    Args:
        line_name: 线路名称（如 C5336）
        training_data_dir: 训练数据根目录
    """
    line_dir = os.path.join(training_data_dir, line_name)
    if not os.path.exists(line_dir):
        raise FileNotFoundError(f'线路目录不存在: {line_dir}')
    if not os.path.isdir(line_dir):
        raise ValueError(f'路径不是目录: {line_dir}')

    return read_test_samples(line_dir, line_name)


def load_all_training_samples_from_line_id(line_id: str, line_name: str) -> Dict:
    """
    从新的线路目录结构加载**所有**训练数据（用于多样本训练）
    
    Args:
        line_id: 线路ID（UUID）
        line_name: 线路名称
    
    Returns:
        {
            'samples': List[Dict],  # 每个样本包含 {bus_data, ub, filename}
            'config': Dict          # 公共配置 {branch_data, voltage_limits, key_nodes, tunable_q_nodes}
        }
    """
    # 导入 line_service
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
    from line_service import line_service
    
    # 获取线路信息
    line_data = line_service.get_line(line_id)
    if not line_data:
        raise FileNotFoundError(f'线路不存在: {line_id}')
    
    # 检查必需的配置数据
    if not line_data.get('lineParamsData'):
        raise ValueError('线路缺少支路数据（lineParamsData）')
    if not line_data.get('voltageLimitData'):
        raise ValueError('线路缺少电压上下限数据（voltageLimitData）')
    if not line_data.get('keyNodesData'):
        raise ValueError('线路缺少关键节点数据（keyNodesData）')
    if not line_data.get('adjustablePvData'):
        raise ValueError('线路缺少可调光伏数据（adjustablePvData）')
    
    # 获取线路目录
    line_data_dir = line_service._get_line_dir(line_id)
    train_dir = os.path.join(line_data_dir, 'train')
    
    # 获取所有训练样本文件
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'训练样本目录不存在: {train_dir}')
        
    sample_files = glob.glob(os.path.join(train_dir, '*.xlsx'))
    if not sample_files:
        raise FileNotFoundError(f'未找到训练样本文件: {train_dir}/*.xlsx')
        
    # 读取所有样本
    samples = []
    for file_path in sample_files:
        try:
            bus_array = _read_bus_matrix(file_path)
            df_slack = pd.read_excel(file_path, sheet_name="slack", header=None)
            ub = _extract_first_float(df_slack)
            
            samples.append({
                'bus_data': bus_array,
                'ub': ub,
                'filename': os.path.basename(file_path)
            })
        except Exception as e:
            print(f"警告：读取样本 {file_path} 失败: {e}")
            continue
            
    if not samples:
        raise ValueError("没有可用的训练样本")
    
    # 读取配置数据（这部分与 load_training_data_from_line_id 相同）
    branch_data = np.array(line_data['lineParamsData'], dtype=float)
    
    voltage_limit_data = line_data['voltageLimitData']
    if not isinstance(voltage_limit_data, list) or len(voltage_limit_data) != 2:
        raise ValueError('电压上下限数据格式错误')
    voltage_lower = float(voltage_limit_data[0])
    voltage_upper = float(voltage_limit_data[1])
    
    key_nodes_data = line_data['keyNodesData']
    key_nodes = [int(node) - 1 for node in key_nodes_data]
    
    adjustable_pv_data = line_data['adjustablePvData']
    tunable_q_nodes_config = []
    
    # 注意：这里需要一个参考的Bus数据来计算Q限制，我们可以用第一个样本的Bus数据
    # 因为不同样本的P可能有变化，导致Q限值变化，但配置中的容量是不变的
    # 在实际训练中，每个样本应该重新计算Q限值，这里只返回配置信息
    
    first_bus_data = samples[0]['bus_data']
    
    for pv in adjustable_pv_data:
        node_num = int(pv[0])
        capacity = float(pv[1])
        node_name = str(pv[2]) if len(pv) > 2 else f"节点{node_num}"
        
        node_index = node_num - 1
        
        # 保存原始配置，具体limit在环境reset时根据当前样本的P计算
        tunable_q_nodes_config.append({
            'node_index': node_index,
            'capacity': capacity,
            'node_name': node_name
        })

    return {
        'samples': samples,
        'config': {
            'branch_data': branch_data,
            'voltage_limits': (voltage_lower, voltage_upper),
            'key_nodes': key_nodes,
            'tunable_q_nodes_config': tunable_q_nodes_config # 这里返回配置而不是计算好的值
        }
    }

