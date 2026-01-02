"""
TD3强化学习优化相关路由
主文件：注册所有子模块的路由
"""
from flask import Blueprint

# 创建主蓝图
td3_optimize_bp = Blueprint('td3_optimize', __name__)

# 从各个子模块导入路由函数
from .td3_optimize_routes import td3_optimize, td3_batch_optimize
from .td3_model_routes import get_td3_models, get_td3_results, get_td3_result_detail, get_td3_template
from .td3_training_routes import start_training, get_training_status, stop_training
from .td3_sample_routes import upload_training_samples, get_training_samples, get_sample_detail
from .td3_agent_routes import get_trained_agents, get_agent_detail, delete_agent
from .td3_data_routes import upload_powerdata_archive, get_powerdata_lines, get_upload_task_status

# 注册所有路由到主蓝图
td3_optimize_bp.add_url_rule('/api/td3-optimize/optimize', 'td3_optimize', td3_optimize, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/batch', 'td3_batch_optimize', td3_batch_optimize, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/models', 'get_td3_models', get_td3_models, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/results', 'get_td3_results', get_td3_results, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/results/<filename>', 'get_td3_result_detail', get_td3_result_detail, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/template', 'get_td3_template', get_td3_template, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/train', 'start_training', start_training, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/training-status', 'get_training_status', get_training_status, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/stop-training', 'stop_training', stop_training, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/upload-samples', 'upload_training_samples', upload_training_samples, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/samples', 'get_training_samples', get_training_samples, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/samples/<filename>', 'get_sample_detail', get_sample_detail, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/agents/<line_name>', 'get_trained_agents', get_trained_agents, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/agents/<line_name>/<model_name>', 'get_agent_detail', get_agent_detail, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/agents/<line_name>/<model_name>', 'delete_agent', delete_agent, methods=['DELETE'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/upload-powerdata', 'upload_powerdata_archive', upload_powerdata_archive, methods=['POST'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/powerdata-lines', 'get_powerdata_lines', get_powerdata_lines, methods=['GET'])
td3_optimize_bp.add_url_rule('/api/td3-optimize/upload-task/<task_id>', 'get_upload_task_status', get_upload_task_status, methods=['GET'])
