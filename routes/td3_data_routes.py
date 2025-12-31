"""
TD3数据上传相关路由
"""
from routes.td3_config import TRAINING_DATA_DIR
from routes.powerdata_upload import (
    handle_upload_powerdata,
    get_powerdata_lines as get_lines_handler,
    get_upload_task_status as get_task_status_handler
)


def upload_powerdata_archive():
    """
    上传训练数据压缩包（zip或tar格式）
    """
    return handle_upload_powerdata(TRAINING_DATA_DIR)


def get_powerdata_lines():
    """获取已上传的线路数据列表"""
    return get_lines_handler(TRAINING_DATA_DIR)


def get_upload_task_status(task_id):
    """获取上传任务状态"""
    return get_task_status_handler(task_id)

