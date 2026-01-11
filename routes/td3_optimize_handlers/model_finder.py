"""
TD3模型路径查找模块
"""
import os
from routes.td3_config import TRAINING_DATA_DIR
from lib.line_service import line_service
from routes.td3_data_reader import read_test_samples


def find_model_path(model_path, line_name):
    """
    查找模型文件路径

    Args:
        model_path: 模型文件名
        line_name: 线路名称（必填，用于在指定线路的agent目录中查找）

    Returns:
        模型完整路径，如果未找到则返回None
    """
    # 只在指定线路的 TRAINING_DATA_DIR/{line_name}/agent/ 中查找
    if not line_name:
        return None

    model_full_path = os.path.join(TRAINING_DATA_DIR, line_name, 'agent', model_path)
    if os.path.exists(model_full_path):
        return model_full_path

    return None


def find_model_path_by_line_id(model_path, line_id):
    """
    通过线路ID查找模型文件路径（从 /api/td3-optimize/train 接口训练完成后的模型文件）

    Args:
        model_path: 模型文件名
        line_id: 线路ID（UUID）

    Returns:
        模型完整路径，如果未找到则返回None
    """
    if not line_id:
        return None

    # 从 line_service 获取线路目录
    line_dir = line_service._get_line_dir(line_id)
    model_full_path = os.path.join(line_dir, 'agent', model_path)

    if os.path.exists(model_full_path):
        return model_full_path

    return None


def load_test_samples_by_line_id(line_id):
    """
    通过线路ID加载测试样本（从 /api/lines/<line_id>/test-sample 接口上传的测试样本目录）

    Args:
        line_id: 线路ID（UUID）

    Returns:
        测试样本列表
    """
    if not line_id:
        raise ValueError('line_id 不能为空')

    # 从 line_service 获取线路目录
    line_dir = line_service._get_line_dir(line_id)

    # 使用 read_test_samples 读取测试样本（line_name 参数在此函数中未使用）
    return read_test_samples(line_dir, None)
