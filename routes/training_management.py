"""
训练管理模块
"""
import os
import json
from datetime import datetime
from flask import jsonify


def get_training_status_info(training_status):
    """
    获取训练状态

    Args:
        training_status: 训练状态字典

    Returns:
        Flask response
    """
    try:
        # 复制状态字典，处理 Infinity 值（JSON 不支持 Infinity）
        status_copy = training_status.copy()
        
        # 将 Infinity 转换为 None 或一个很大的数字，前端会处理
        if status_copy.get('best_loss_rate') == float('inf'):
            status_copy['best_loss_rate'] = None  # 前端会显示为 '-'
        
        return jsonify({
            'status': 'success',
            'data': status_copy
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取训练状态失败: {str(e)}'
        }), 500


def stop_training_task(training_status):
    """
    停止训练任务

    Args:
        training_status: 训练状态字典

    Returns:
        Flask response
    """
    try:
        if not training_status['is_training']:
            return jsonify({
                'status': 'error',
                'message': '当前没有正在进行的训练任务'
            }), 400

        training_status['is_training'] = False
        training_status['message'] = '训练已停止'

        return jsonify({
            'status': 'success',
            'message': '训练停止请求已发送'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'停止训练失败: {str(e)}'
        }), 500
