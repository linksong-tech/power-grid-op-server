"""
TD3智能体（模型）管理相关路由
"""
from flask import jsonify
import json
import os
from datetime import datetime
from routes.td3_config import MODELS_DIR, get_lib_dir
from routes.agent_management import get_agents_list, delete_agent_by_name


def get_trained_agents():
    """获取历史训练的智能体（模型）列表"""
    lib_dir = get_lib_dir()
    return get_agents_list(MODELS_DIR, lib_dir)


def get_agent_detail(model_name):
    """获取智能体详细信息"""
    try:
        # 查找模型文件
        model_file = None
        for directory in [MODELS_DIR, get_lib_dir()]:
            potential_path = os.path.join(directory, f'{model_name}.pth')
            if os.path.exists(potential_path):
                model_file = potential_path
                break
        
        if not model_file:
            return jsonify({
                'status': 'error',
                'message': '模型文件不存在'
            }), 404
        
        file_stat = os.stat(model_file)
        
        # 读取训练历史
        history_file = model_file.replace('.pth', '_history.json')
        training_history = None
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                training_history = json.load(f)
        
        # 读取模型元信息
        meta_file = model_file.replace('.pth', '_meta.json')
        meta_info = None
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
        
        return jsonify({
            'status': 'success',
            'data': {
                'model_name': model_name,
                'filename': f'{model_name}.pth',
                'size': file_stat.st_size,
                'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'training_history': training_history,
                'meta_info': meta_info
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取智能体详情失败: {str(e)}'
        }), 500


def delete_agent(model_name):
    """删除智能体（模型）"""
    return delete_agent_by_name(model_name, MODELS_DIR)

