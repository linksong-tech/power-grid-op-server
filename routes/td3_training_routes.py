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

from td3_train_service_numpy import train_td3_model
from routes.td3_config import training_status
from routes.training_management import get_training_status_info, stop_training_task
from routes.td3_data_reader import load_training_data_from_line_id
from lib.line_service import line_service

# 模型文件名前缀
MODEL_FILENAME_PREFIX = "M1"


def start_training():
    """
    开始TD3模型训练
    
    请求体格式（新版本）:
    {
        "line_id": "uuid-string",  // 线路ID（必填）
        "parameters": {
            "max_episodes": 800,    // 最大训练轮次（可选，默认800）
            "max_steps": 3,          // 每轮最大步数（可选，默认3）
            "SB": 10,                 // 基准功率MVA（可选，默认10）
            "pr": 1e-6                // 潮流收敛精度（可选，默认1e-6）
        }
    }
    
    说明：
    - 所有训练数据（busData, branchData, voltageLimits, keyNodes, tunableNodes, UB）
      将从线路目录中自动读取
    - 线路目录路径: data/line_data/{line_id}/
    - 训练样本路径: data/line_data/{line_id}/train/
    - 模型保存路径: data/line_data/{line_id}/agent/
    - 模型保存格式: {MODEL_FILENAME_PREFIX}_MMDD_HHMMSS.pth
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
        if 'line_id' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需参数: line_id'
            }), 400
        
        line_id = data['line_id']
        if not line_id or not isinstance(line_id, str):
            return jsonify({
                'status': 'error',
                'message': 'line_id 必须是有效的字符串'
            }), 400
        
        # 获取线路信息
        line_data = line_service.get_line(line_id)
        if not line_data:
            return jsonify({
                'status': 'error',
                'message': f'线路不存在: {line_id}'
            }), 404
        
        line_name = line_data['name']
        
        # 获取训练参数
        params = data.get('parameters', {})
        max_episodes = params.get('max_episodes', 800)
        max_steps = params.get('max_steps', 3)
        sb = params.get('SB', 10.0)
        pr = params.get('pr', 1e-6)
        
        # 验证训练参数
        if not isinstance(max_episodes, int) or max_episodes <= 0:
            return jsonify({
                'status': 'error',
                'message': 'max_episodes 必须是正整数'
            }), 400
        
        if not isinstance(max_steps, int) or max_steps <= 0:
            return jsonify({
                'status': 'error',
                'message': 'max_steps 必须是正整数'
            }), 400
        
        # 初始化训练状态 - 直接替换所有字段，避免残留数据
        training_status['is_training'] = True
        training_status['line_id'] = line_id
        training_status['line_name'] = line_name
        training_status['model_name'] = MODEL_FILENAME_PREFIX
        training_status['current_episode'] = 0
        training_status['total_episodes'] = max_episodes
        training_status['current_reward'] = 0
        training_status['current_loss_rate'] = 0
        training_status['best_loss_rate'] = float('inf')
        training_status['message'] = '正在加载训练数据...'
        training_status['start_time'] = datetime.now().isoformat()
        training_status['end_time'] = None
        training_status['result'] = None
        training_status['error'] = None
        training_status['training_history'] = {
            'rewards': [],
            'loss_rates': []
        }
        training_status['logs'] = []
        
        # 定义进度回调函数
        def progress_callback(episode, reward, loss_rate):
            training_status['current_episode'] = episode
            training_status['current_reward'] = reward
            training_status['current_loss_rate'] = loss_rate
            if loss_rate < training_status['best_loss_rate']:
                training_status['best_loss_rate'] = loss_rate
            training_status['message'] = f'训练中: Episode {episode}/{max_episodes}'
            
            # 更新训练历史数据
            training_status['training_history']['rewards'].append(reward)
            training_status['training_history']['loss_rates'].append(loss_rate)
            
            # 添加训练日志（对象格式）
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'episode': episode,
                'reward': round(reward, 2),
                'loss_rate': round(loss_rate, 4),
                'best_loss_rate': round(training_status['best_loss_rate'], 4)
            }
            training_status['logs'].append(log_entry)
            
            # 限制日志数量，只保留最近1000条
            if len(training_status['logs']) > 1000:
                training_status['logs'] = training_status['logs'][-1000:]
        
        # 在后台线程中执行训练
        def train_in_background():
            global training_status
            try:
                # 加载训练数据（从新的线路目录读取）
                training_status['message'] = '正在读取训练数据...'
                print(f"\n=== 开始加载训练数据 ===")
                print(f"线路ID: {line_id}")
                print(f"线路名称: {line_name}")
                
                training_data = load_training_data_from_line_id(line_id, line_name)
                
                # 提取数据
                bus_data = training_data['bus_data']
                branch_data = training_data['branch_data']
                voltage_limits = training_data['voltage_limits']
                key_nodes = training_data['key_nodes']
                tunable_q_nodes = training_data['tunable_q_nodes']
                ub = training_data['ub']
                
                # 打印配置信息
                print(f"电压上下限: [{voltage_limits[0]:.2f}kV, {voltage_limits[1]:.2f}kV]")
                print(f"关键节点索引: {key_nodes} (共{len(key_nodes)}个)")
                print(f"可调节无功节点: {[node[3] for node in tunable_q_nodes]} (共{len(tunable_q_nodes)}个)")
                print(f"基准电压UB: {ub:.2f}kV")
                print(f"基准功率SB: {sb:.2f}MVA")
                print(f"Bus节点数: {bus_data.shape[0]}")
                print(f"支路数: {branch_data.shape[0]}")
                
                training_status['message'] = '训练数据加载完成，开始训练...'

                # 构建模型保存路径: data/line_data/{line_id}/agent/
                line_dir = line_service._get_line_dir(line_id)
                agent_dir = os.path.join(line_dir, 'agent')
                os.makedirs(agent_dir, exist_ok=True)

                # 生成带时间戳的模型文件名
                timestamp = datetime.now().strftime('%m%d_%H%M%S')
                model_filename = f'{MODEL_FILENAME_PREFIX}_{timestamp}.pth'
                model_save_path = os.path.join(agent_dir, model_filename)
                
                # 执行训练
                result = train_td3_model(
                    bus_data=bus_data,
                    branch_data=branch_data,
                    voltage_limits=voltage_limits,
                    key_nodes=key_nodes,
                    tunable_q_nodes=tunable_q_nodes,
                    ub=ub,
                    sb=sb,
                    max_episodes=max_episodes,
                    max_steps=max_steps,
                    model_save_path=model_save_path,
                    pr=pr,
                    progress_callback=progress_callback
                )
                
                training_status['is_training'] = False
                training_status['message'] = '训练完成'
                training_status['result'] = result
                training_status['end_time'] = datetime.now().isoformat()
                
            except FileNotFoundError as e:
                training_status['is_training'] = False
                training_status['message'] = f'训练数据文件不存在: {str(e)}'
                training_status['error'] = str(e)
            except ValueError as e:
                training_status['is_training'] = False
                training_status['message'] = f'训练数据格式错误: {str(e)}'
                training_status['error'] = str(e)
            except Exception as e:
                training_status['is_training'] = False
                training_status['message'] = f'训练失败: {str(e)}'
                training_status['error'] = str(e)
                print(traceback.format_exc())
        
        # 启动训练线程
        train_thread = threading.Thread(target=train_in_background)
        train_thread.daemon = True
        train_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '训练任务已启动',
            'data': {
                'line_id': line_id,
                'line_name': line_name,
                'model_name': MODEL_FILENAME_PREFIX,
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
