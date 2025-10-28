import os
import json
from typing import List, Tuple, Optional


def get_latest_oc_source_file(directory: str = '/app/hjpq') -> Optional[str]:
    """
    获取最新的运行工况数据源文件路径
    
    Args:
        directory: 数据源目录路径
    
    Returns:
        最新文件的完整路径，如果未找到则返回None
    
    Raises:
        FileNotFoundError: 目录不存在
        Exception: 其他文件操作错误
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f'数据源目录不存在: {directory}')
    
    # 查找匹配的文件
    json_files = []
    for filename in os.listdir(directory):
        if filename.startswith('hjdata_') and filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            json_files.append((filepath, filename))
    
    if not json_files:
        return None
    
    # 按文件名排序，选择最新的文件
    json_files.sort(key=lambda x: x[1], reverse=True)
    return json_files[0][0]


def read_oc_source_data(filepath: str) -> List[dict]:
    """
    读取运行工况数据源文件
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        解析后的数据列表
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析失败
        ValueError: 数据格式错误
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'数据文件不存在: {filepath}')
    
    # 读取JSON文件
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据格式
    if not isinstance(data, list):
        raise ValueError('数据格式错误：期望数组格式')
    
    # 验证每个元素
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'数据格式错误：第{i+1}个数组元素应为对象')
        
        # 验证必需字段
        if 'id' not in item or 'p' not in item or 'q' not in item:
            raise ValueError(f'数据格式错误：第{i+1}个元素缺少必需字段 id, p, q')
        
        # 验证数据类型
        try:
            float(item['id'])
            float(item['p'])
            float(item['q'])
        except (ValueError, TypeError):
            raise ValueError(f'数据格式错误：第{i+1}个元素的 id, p, q 应为数字')
    
    return data


def convert_to_flat_array(data: List[dict]) -> List[float]:
    """
    将运行工况数据转换为扁平数组
    
    Args:
        data: 原始数据列表，格式为 [{id, p, q}, ...]
    
    Returns:
        扁平数组，格式为 [id1, p1, q1, id2, p2, q2, ...]
    """
    flat_data = []
    for item in data:
        flat_data.append(float(item['id']))
        flat_data.append(float(item['p']))
        flat_data.append(float(item['q']))
    return flat_data


def get_oc_source_flat_data(directory: str = '/app/hjpq') -> Tuple[Optional[List[float]], Optional[str]]:
    """
    获取最新的运行工况数据并转换为扁平数组
    
    Args:
        directory: 数据源目录路径
    
    Returns:
        (扁平数组数据, 错误消息) 元组
        - 成功时返回 (数据, None)
        - 失败时返回 (None, 错误消息)
    """
    try:
        # 获取最新文件
        latest_file = get_latest_oc_source_file(directory)
        if latest_file is None:
            return None, '未找到数据文件'
        
        # 读取数据
        data = read_oc_source_data(latest_file)
        
        # 转换为扁平数组
        flat_data = convert_to_flat_array(data)
        
        return flat_data, None
    
    except FileNotFoundError as e:
        return None, str(e)
    except json.JSONDecodeError as e:
        return None, f'JSON解析失败: {str(e)}'
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f'获取数据源失败: {str(e)}'

