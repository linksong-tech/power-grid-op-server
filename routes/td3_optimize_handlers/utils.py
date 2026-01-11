"""
TD3优化工具函数模块
"""
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import os


def get_timezone():
    """获取时区配置，默认为 Asia/Shanghai"""
    tz_name = os.environ.get('TZ', 'Asia/Shanghai')
    return ZoneInfo(tz_name)


def sanitize_for_json(value):
    """
    将 Python 值转换为 JSON 安全的值
    - float('inf') -> None (JSON 中为 null)
    - float('-inf') -> None
    - float('nan') -> None
    """
    if isinstance(value, float):
        if np.isinf(value) or np.isnan(value):
            return None
    return value


def format_datetime(dt_str):
    """
    将日期时间字符串转换为 YYYY-MM-DD HH:mm 或 YYYY-MM-DD HH:mm:ss 格式
    支持多种输入格式：
    - ISO格式：2026-01-06T10:30:45
    - 紧凑格式（无秒）：202601061030 -> 2026-01-06 10:30
    - 紧凑格式（有秒）：20260106103045 -> 2026-01-06 10:30:45
    - 已格式化：2026-01-06 10:30:45

    Args:
        dt_str: 日期时间字符串或None

    Returns:
        格式化后的日期时间字符串，如果输入为None则返回None
    """
    if not dt_str:
        return None

    # 如果已经是目标格式，直接返回
    if isinstance(dt_str, str) and len(dt_str) in [16, 19] and dt_str[10] == ' ':
        return dt_str

    try:
        # 尝试解析紧凑格式 YYYYMMDDHHmm (12位，无秒) -> 格式化为 YYYY-MM-DD HH:mm
        if isinstance(dt_str, str) and len(dt_str) == 12 and dt_str.isdigit():
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
            return dt.strftime("%Y-%m-%d %H:%M")

        # 尝试解析紧凑格式 YYYYMMDDHHmmss (14位，有秒) -> 格式化为 YYYY-MM-DD HH:mm:ss
        if isinstance(dt_str, str) and len(dt_str) == 14 and dt_str.isdigit():
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        # 尝试解析ISO格式
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        # 如果解析失败，返回原始字符串
        return dt_str


def build_tunable_q_nodes(tunable_nodes, bus_data: np.ndarray):
    """
    支持两种格式：
    - [node_index, q_min, q_max, name]：直接使用（node_index 为 0-based）
    - [node_no, capacity, name]：根据 bus_data 计算 q_min/q_max（node_no 为 1-based）
    """
    built = []
    for node in tunable_nodes:
        if not isinstance(node, (list, tuple)):
            raise ValueError("tunableNodes 格式错误：元素必须为数组")
        if len(node) == 4:
            node_idx = int(node[0])
            q_min = float(node[1])
            q_max = float(node[2])
            name = str(node[3])
            built.append((node_idx, q_min, q_max, name))
            continue
        if len(node) == 3:
            node_no = int(node[0])
            capacity = float(node[1])
            name = str(node[2])
            node_idx = node_no - 1
            if node_idx < 0 or node_idx >= bus_data.shape[0]:
                raise ValueError(f"tunableNodes 节点号超出范围: {node_no}")
            p_current = float(abs(bus_data[node_idx, 1]))
            q_max = float(np.sqrt(max(0.0, capacity**2 - p_current**2)))
            q_min = -q_max
            built.append((node_idx, q_min, q_max, name))
            continue
        raise ValueError("tunableNodes 格式错误：仅支持长度为3或4的数组")
    return built
