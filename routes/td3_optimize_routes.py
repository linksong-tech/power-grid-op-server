"""
TD3优化执行相关路由
"""
from flask import jsonify, request
import json
import numpy as np
import os
import sys
from datetime import datetime
import traceback

# 添加lib目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from td3_inference_service import optimize_reactive_power, batch_optimize
from routes.td3_config import MODELS_DIR, RESULTS_DIR, get_lib_dir


def find_model_path(model_path):
    """查找模型文件路径"""
    model_full_path = os.path.join(MODELS_DIR, model_path)
    if not os.path.exists(model_full_path):
        model_full_path = os.path.join(get_lib_dir(), model_path)
        if not os.path.exists(model_full_path):
            return None
    return model_full_path


def td3_optimize():
    """
    执行TD3强化学习无功优化
    
    请求体格式:
    {
        "busData": [[节点号, 有功P, 无功Q], ...],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "modelPath": "模型文件名",
        "parameters": {
            "UB": 10.38,
            "SB": 10
        }
    }
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_fields = ['busData', 'branchData', 'voltageLimits', 'keyNodes', 'tunableNodes', 'modelPath']
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
        model_path = data['modelPath']
        
        # 获取物理参数
        params = data.get('parameters', {})
        ub = params.get('UB', 10.38)
        sb = params.get('SB', 10)
        
        # 检查模型文件是否存在
        model_full_path = find_model_path(model_path)
        if not model_full_path:
            return jsonify({
                'status': 'error',
                'message': f'模型文件不存在: {model_path}'
            }), 404
        
        # 执行TD3优化
        result = optimize_reactive_power(
            bus_data=bus_data,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_nodes,
            model_path=model_full_path,
            ub=ub,
            sb=sb
        )
        
        if not result['success']:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'TD3优化失败')
            }), 500
        
        # 保存优化结果
        result_file = os.path.join(RESULTS_DIR, f'td3_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            save_result = result.copy()
            if 'optimized_voltages' in save_result:
                save_result['optimized_voltages'] = save_result['optimized_voltages'].tolist()
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        # 返回结果（转换numpy数组）
        response_result = result.copy()
        if 'optimized_voltages' in response_result:
            response_result['optimized_voltages'] = response_result['optimized_voltages'].tolist()
        
        return jsonify({
            'status': 'success',
            'data': response_result
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'TD3优化失败: {str(e)}'
        }), 500


def td3_batch_optimize():
    """
    批量执行TD3优化
    
    请求体格式:
    {
        "testSamples": [
            {
                "time": "2024-12-29 08:00",
                "ub": 10.0,
                "bus": [节点号1, P1, Q1, 节点号2, P2, Q2, ...]
            },
            ...
        ],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "modelPath": "模型文件名",
        "parameters": {
            "SB": 10
        }
    }
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_fields = ['testSamples', 'branchData', 'voltageLimits', 'keyNodes', 'tunableNodes', 'modelPath']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必需参数: {field}'
                }), 400
        
        # 获取参数
        test_samples = data['testSamples']
        branch_data = np.array(data['branchData'])
        voltage_limits = tuple(data['voltageLimits'])
        key_nodes = data['keyNodes']
        tunable_nodes = [tuple(node) for node in data['tunableNodes']]
        model_path = data['modelPath']
        
        # 获取物理参数
        params = data.get('parameters', {})
        sb = params.get('SB', 10)
        
        # 检查模型文件是否存在
        model_full_path = find_model_path(model_path)
        if not model_full_path:
            return jsonify({
                'status': 'error',
                'message': f'模型文件不存在: {model_path}'
            }), 404
        
        # 执行批量优化
        result = batch_optimize(
            test_samples=test_samples,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_nodes,
            model_path=model_full_path,
            sb=sb
        )
        
        if not result['success']:
            return jsonify({
                'status': 'error',
                'message': '批量优化失败'
            }), 500
        
        # 保存批量优化结果
        result_file = os.path.join(RESULTS_DIR, f'td3_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表
            save_result = result.copy()
            for r in save_result.get('results', []):
                if 'optimized_voltages' in r and hasattr(r['optimized_voltages'], 'tolist'):
                    r['optimized_voltages'] = r['optimized_voltages'].tolist()
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        # 返回结果（转换numpy数组）
        response_result = result.copy()
        for r in response_result.get('results', []):
            if 'optimized_voltages' in r and hasattr(r['optimized_voltages'], 'tolist'):
                r['optimized_voltages'] = r['optimized_voltages'].tolist()
        
        return jsonify({
            'status': 'success',
            'data': response_result
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'批量优化失败: {str(e)}'
        }), 500

