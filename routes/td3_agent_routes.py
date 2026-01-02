"""
TD3智能体（模型）管理相关路由
"""
from flask import jsonify, request
import json
import os
from datetime import datetime
from routes.td3_config import TRAINING_DATA_DIR
from routes.agent_management import get_agents_list, delete_agent_by_name


def get_trained_agents(line_name):
    """获取指定线路的历史训练智能体（模型）列表"""
    return get_agents_list(TRAINING_DATA_DIR, line_name)


def get_agent_detail(line_name, model_name):
    """获取智能体详细信息"""
    try:
        # 在指定线路的agent目录中查找模型文件
        agent_dir = os.path.join(TRAINING_DATA_DIR, line_name, 'agent')
        model_file = os.path.join(agent_dir, f'{model_name}.npz')

        if not os.path.exists(model_file):
            return jsonify({
                'status': 'error',
                'message': '模型文件不存在'
            }), 404
        
        file_stat = os.stat(model_file)
        
        # 读取训练历史
        history_file = model_file.replace('.npz', '_history.json')
        training_history = None
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                training_history = json.load(f)
        
        # 读取模型元信息
        meta_file = model_file.replace('.npz', '_meta.json')
        meta_info = None
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
        
        return jsonify({
            'status': 'success',
            'data': {
                'model_name': model_name,
                'filename': f'{model_name}.npz',
                'line_name': line_name,
                'size': file_stat.st_size,
                'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'training_history': training_history,
                'meta_info': meta_info,
                'location': f'{line_name}/agent'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取智能体详情失败: {str(e)}'
        }), 500


def delete_agent(line_name, model_name):
    """删除智能体（模型）"""
    return delete_agent_by_name(model_name, line_name, TRAINING_DATA_DIR)

