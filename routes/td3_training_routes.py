"""
TD3训练相关路由
"""
from flask import jsonify, request
import numpy as np
import os
import sys
from datetime import datetime
import threading
import traceback

# 添加lib目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from td3_train_service import train_td3_model
from routes.td3_config import MODELS_DIR, training_status
from routes.training_management import get_training_status_info, stop_training_task


def start_training():
    """
    开始TD3模型训练
    
    请求体格式:
    {
        "busData": [[节点号, 有功P, 无功Q], ...],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "parameters": {
            "UB": 10.38,
            "SB": 10,
            "max_episodes": 800,
            "max_steps": 3,
            "model_name": "my_model"
        }
    }
    """
    global training_status
    
    try:
        # 检查是否正在训练
        if training_status['is_training']:
            return jsonify({
                'status': 'error',
                'message': '已有训练任务正在进行中'
            }), 400
        
        data = request.get_json()
        
        # 验证必需参数
        required_fields = ['busData', 'branchData', 'voltageLimits', 'keyNodes', 'tunableNodes']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必需参数: {field}'
                }), 400
        
        # 获取参数
        bus_data = np.array(data['busData'])
        branch_data = np.array(data['branchData'])
        voltage_limits = tuple(data['voltageLimits'])
        key_nodes = data['keyNodes']
        tunable_nodes = [tuple(node) for node in data['tunableNodes']]
        
        # 获取训练参数
        params = data.get('parameters', {})
        ub = params.get('UB', 10.38)
        sb = params.get('SB', 10)
        max_episodes = params.get('max_episodes', 800)
        max_steps = params.get('max_steps', 3)
        model_name = params.get('model_name', f'td3_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # 初始化训练状态
        training_status.update({
            'is_training': True,
            'current_episode': 0,
            'total_episodes': max_episodes,
            'current_reward': 0,
            'current_loss_rate': 0,
            'best_loss_rate': float('inf'),
            'message': '训练初始化中...',
            'start_time': datetime.now().isoformat()
        })
        
        # 定义进度回调函数
        def progress_callback(episode, reward, loss_rate):
            training_status['current_episode'] = episode
            training_status['current_reward'] = reward
            training_status['current_loss_rate'] = loss_rate
            if loss_rate < training_status['best_loss_rate']:
                training_status['best_loss_rate'] = loss_rate
            training_status['message'] = f'训练中: Episode {episode}/{max_episodes}'
        
        # 在后台线程中执行训练
        def train_in_background():
            global training_status
            try:
                model_save_path = os.path.join(MODELS_DIR, f'{model_name}.pth')
                
                result = train_td3_model(
                    bus_data=bus_data,
                    branch_data=branch_data,
                    voltage_limits=voltage_limits,
                    key_nodes=key_nodes,
                    tunable_q_nodes=tunable_nodes,
                    ub=ub,
                    sb=sb,
                    max_episodes=max_episodes,
                    max_steps=max_steps,
                    model_save_path=model_save_path,
                    progress_callback=progress_callback
                )
                
                training_status['is_training'] = False
                training_status['message'] = '训练完成'
                training_status['result'] = result
                training_status['end_time'] = datetime.now().isoformat()
                
            except Exception as e:
                training_status['is_training'] = False
                training_status['message'] = f'训练失败: {str(e)}'
                training_status['error'] = str(e)
        
        # 启动训练线程
        train_thread = threading.Thread(target=train_in_background)
        train_thread.daemon = True
        train_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '训练任务已启动',
            'data': {
                'model_name': model_name,
                'total_episodes': max_episodes
            }
        })
        
    except Exception as e:
        training_status['is_training'] = False
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'启动训练失败: {str(e)}'
        }), 500


def get_training_status():
    """获取训练状态"""
    return get_training_status_info(training_status)


def stop_training():
    """停止训练"""
    return stop_training_task(training_status)

