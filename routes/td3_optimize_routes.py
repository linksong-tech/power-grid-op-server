"""
TD3优化执行相关路由
"""
import os
import sys

# 添加lib目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

# 导入重构后的模块
from routes.td3_optimize_handlers import (
    td3_optimize,
    td3_batch_optimize,
    get_td3_batch_status,
    get_performance_thresholds,
    update_performance_thresholds,
    get_evaluation_report,
)

# 导出所有路由处理函数
__all__ = [
    'td3_optimize',
    'td3_batch_optimize',
    'get_td3_batch_status',
    'get_performance_thresholds',
    'update_performance_thresholds',
    'get_evaluation_report',
]
