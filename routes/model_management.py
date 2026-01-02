"""
模型和结果管理模块
"""
import os
import json
from datetime import datetime
from flask import jsonify


def get_models_list(training_data_dir):
    """
    获取可用的TD3模型列表

    Args:
        training_data_dir: 训练数据目录，扫描各线路的agent目录

    Returns:
        Flask response
    """
    try:
        models = []

        # 扫描training_data_dir下各线路的agent目录
        if training_data_dir and os.path.exists(training_data_dir):
            for line_name in os.listdir(training_data_dir):
                agent_dir = os.path.join(training_data_dir, line_name, 'agent')
                if os.path.isdir(agent_dir):
                    for filename in os.listdir(agent_dir):
                        if filename.endswith('.pth'):
                            filepath = os.path.join(agent_dir, filename)
                            file_stat = os.stat(filepath)
                            models.append({
                                'filename': filename,
                                'line_name': line_name,
                                'size': file_stat.st_size,
                                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                                'location': f'{line_name}/agent'
                            })

        # 按修改时间排序
        models.sort(key=lambda x: x.get('modified_time', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'data': models
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取模型列表失败: {str(e)}'
        }), 500


def get_results_list(results_dir):
    """
    获取TD3优化结果列表

    Args:
        results_dir: 结果目录

    Returns:
        Flask response
    """
    try:
        results = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith('td3_result_') and filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        results.append({
                            'filename': filename,
                            'timestamp': filename.replace('td3_result_', '').replace('.json', ''),
                            'initial_loss_rate': result_data.get('initial_loss_rate'),
                            'optimized_loss_rate': result_data.get('optimized_loss_rate'),
                            'loss_reduction': result_data.get('loss_reduction'),
                            'voltage_violations': result_data.get('voltage_violations')
                        })

        # 按时间戳排序
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'status': 'success',
            'data': results
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取TD3结果列表失败: {str(e)}'
        }), 500

