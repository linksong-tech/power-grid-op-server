"""
模型和结果管理模块
"""
import os
import json
from datetime import datetime
from flask import jsonify


def get_models_list(models_dir, lib_dir):
    """
    获取可用的TD3模型列表

    Args:
        models_dir: 模型目录
        lib_dir: lib目录

    Returns:
        Flask response
    """
    try:
        models = []

        # 扫描models目录
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pth'):
                    filepath = os.path.join(models_dir, filename)
                    file_stat = os.stat(filepath)
                    models.append({
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })

        # 扫描lib目录
        if os.path.exists(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename.endswith('.pth'):
                    filepath = os.path.join(lib_dir, filename)
                    file_stat = os.stat(filepath)
                    models.append({
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'location': 'lib'
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

