"""
TD3训练样本管理相关路由
"""
from routes.td3_config import TRAINING_SAMPLES_DIR
from routes.sample_management import upload_samples, get_samples_list, get_sample_detail_by_filename


def upload_training_samples():
    """上传训练样本"""
    return upload_samples(TRAINING_SAMPLES_DIR)


def get_training_samples():
    """获取已上传的训练样本列表"""
    return get_samples_list(TRAINING_SAMPLES_DIR)


def get_sample_detail(filename):
    """获取训练样本详情"""
    return get_sample_detail_by_filename(filename, TRAINING_SAMPLES_DIR)

