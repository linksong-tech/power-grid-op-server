import requests
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def fetch_remote_data(
    host: str = "host.docker.internal",
    port: int = 12345,
    timeout: int = 10
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    从宿主机接口获取原始数据
    
    Args:
        host: 宿主机地址
        port: 端口号
        timeout: 请求超时时间（秒）
    
    Returns:
        (响应数据字典, 错误消息) 元组
        - 成功时返回 (数据, None)
        - 失败时返回 (None, 错误消息)
    """
    url = f"http://{host}:{port}/api/query"
    payload = {
        "column": "id,name,value",
        "scn_inst": "1",
        "scn_name": "realtime",
        "sub_scn_inst": "1",
        "sub_scn_name": "scada",
        "table_no": "102001"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # 验证响应格式
        if not isinstance(data, dict):
            return None, "响应格式错误：期望对象格式"
        
        if "data" not in data:
            return None, "响应缺少 data 字段"
        
        if not isinstance(data["data"], list):
            return None, "data 字段应为数组"
        
        return data, None
        
    except requests.exceptions.Timeout:
        return None, f"请求超时（{timeout}秒）"
    except requests.exceptions.ConnectionError:
        return None, f"无法连接到宿主机 {host}:{port}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP错误: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"请求失败: {str(e)}"
    except ValueError as e:
        return None, f"JSON解析失败: {str(e)}"
    except Exception as e:
        return None, f"获取远程数据失败: {str(e)}"


def filter_huangjian_data(data_list: List[Dict]) -> List[Dict]:
    """
    筛选 name 字段包含"黄尖简化"的记录
    
    Args:
        data_list: 原始数据列表
    
    Returns:
        筛选后的数据列表
    """
    filtered = []
    for item in data_list:
        if not isinstance(item, dict):
            continue
        
        name = item.get("name", "")
        if isinstance(name, str) and "黄尖简化" in name:
            filtered.append(item)
    
    return filtered


def merge_pq_data(filtered_data: List[Dict]) -> List[Dict]:
    """
    合并有功无功数据
    
    Args:
        filtered_data: 筛选后的数据列表
    
    Returns:
        合并后的数据列表，格式为 [{"id": id, "p": p, "q": q}, ...]
    """
    # 使用字典按名称前缀分组
    grouped = defaultdict(dict)
    
    for item in filtered_data:
        name = item.get("name", "")
        value = item.get("value", "0")
        
        # 提取前缀（去掉"有功"或"无功"后缀）
        # 例如："黄尖简化1节点有功" -> "黄尖简化1节点"
        name_prefix = None
        is_power = False  # False表示无功
        
        if name.endswith("有功"):
            name_prefix = name[:-2]  # 去掉"有功"
            is_power = True
        elif name.endswith("无功"):
            name_prefix = name[:-2]  # 去掉"无功"
            is_power = False
        else:
            # 不匹配的记录跳过
            continue
        
        # 提取节点编号（从"黄尖简化X"中提取X）
        match = re.search(r'黄尖简化(\d+)', name)
        if not match:
            continue
        
        node_id = int(match.group(1))
        
        # 尝试转换value为float
        try:
            value_float = float(value)
        except (ValueError, TypeError):
            value_float = 0.0
        
        # 存储到分组字典
        if node_id not in grouped:
            grouped[node_id] = {"id": node_id}
        
        if is_power:
            grouped[node_id]["p"] = value_float
        else:
            grouped[node_id]["q"] = value_float
    
    # 转换为列表，只保留同时有p和q的数据
    result = []
    for node_id in sorted(grouped.keys()):
        node_data = grouped[node_id]
        if "p" in node_data and "q" in node_data:
            result.append(node_data)
    
    return result


def convert_to_flat(merged_data: List[Dict]) -> List[float]:
    """
    转换为扁平数组格式
    
    Args:
        merged_data: 合并后的数据列表，格式为 [{"id": id, "p": p, "q": q}, ...]
    
    Returns:
        扁平数组，格式为 [id1, p1, q1, id2, p2, q2, ...]
    """
    flat_data = []
    for item in merged_data:
        flat_data.append(float(item["id"]))
        flat_data.append(float(item["p"]))
        flat_data.append(float(item["q"]))
    return flat_data


def get_remote_oc_source_data(
    host: str = "host.docker.internal",
    port: int = 12345,
    timeout: int = 10
) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    获取远程运行工况数据并转换为扁平数组
    
    Args:
        host: 宿主机地址
        port: 端口号
        timeout: 请求超时时间（秒）
    
    Returns:
        (扁平数组数据, 错误消息) 元组
        - 成功时返回 (数据, None)
        - 失败时返回 (None, 错误消息)
    """
    try:
        # 1. 获取远程数据
        remote_data, error = fetch_remote_data(host, port, timeout)
        if error:
            return None, error
        
        # 2. 筛选数据
        filtered = filter_huangjian_data(remote_data["data"])
        
        if not filtered:
            return None, "未找到包含'黄尖简化'的数据记录"
        
        # 3. 合并有功无功数据
        merged = merge_pq_data(filtered)
        
        if not merged:
            return None, "未能成功配对有功和无功数据"
        
        # 4. 转换为扁平数组
        flat_data = convert_to_flat(merged)
        
        return flat_data, None
        
    except Exception as e:
        return None, f"处理远程数据失败: {str(e)}"

