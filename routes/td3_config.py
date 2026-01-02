"""
TD3优化相关配置和共享状态
"""
import os
import threading

# 配置目录 - 统一收集到 td3_training_data 目录下
BASE_DIR = 'td3_training_data'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAINING_SAMPLES_DIR = os.path.join(BASE_DIR, 'training_samples')
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')

# 确保目录存在
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_SAMPLES_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# 训练状态管理
training_status = {
    'is_training': False,
    'line_name': None,  # 线路名称
    'model_name': None,  # 模型名称
    'current_episode': 0,
    'total_episodes': 0,
    'current_reward': 0,
    'current_loss_rate': 0,
    'best_loss_rate': float('inf'),
    'message': '',
    'start_time': None,
    'end_time': None,
    'result': None,
    'error': None,
    'training_history': {
        'rewards': [],
        'loss_rates': []
    },
    'logs': []  # 训练日志
}

# 批量评估/优化任务状态管理（用于前端轮询）
# job_id -> job_status
batch_jobs = {}
batch_jobs_lock = threading.Lock()


def get_lib_dir():
    """获取lib目录路径"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib')
