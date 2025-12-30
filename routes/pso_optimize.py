"""
PSO优化相关路由
"""
from flask import Blueprint, jsonify, request
import json
import numpy as np
import os
from datetime import datetime
from lib.pso_op import pso_op
from lib.pso_op_v2 import pso_op as pso_op_v2

pso_optimize_bp = Blueprint('pso_optimize', __name__)

# 配置目录
DATA_DIR = 'data'
RESULTS_DIR = 'results'

@pso_optimize_bp.route('/api/pso-optimize/parameters', methods=['POST'])
def save_pso_parameters():
    """保存PSO优化参数"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_fields = ['num_particles', 'max_iter', 'w', 'c1', 'c2', 'v_min', 'v_max']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必需参数: {field}'
                }), 400
        
        # 保存参数到文件
        params_file = os.path.join(DATA_DIR, 'pso_parameters.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': 'PSO参数保存成功',
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'PSO参数保存失败: {str(e)}'
        }), 500

@pso_optimize_bp.route('/api/pso-optimize/parameters', methods=['GET'])
def get_pso_parameters():
    """获取保存的PSO优化参数"""
    try:
        params_file = os.path.join(DATA_DIR, 'pso_parameters.json')
        if os.path.exists(params_file):
            with open(params_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({
                'status': 'success',
                'data': data
            })
        else:
            # 返回默认参数
            default_params = {
                'num_particles': 30,
                'max_iter': 50,
                'w': 0.8,
                'c1': 1.5,
                'c2': 1.5,
                'v_min': 0.95,
                'v_max': 1.05,
                'SB': 10,
                'UB': 10.38,
                'pr': 1e-6
            }
            return jsonify({
                'status': 'success',
                'data': default_params
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取PSO参数失败: {str(e)}'
        }), 500

@pso_optimize_bp.route('/api/pso-optimize/optimize', methods=['POST'])
def pso_optimize():
    """执行PSO无功优化"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'busData' not in data or 'branchData' not in data or 'tunableNodes' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需的节点数据、支路数据或可调节点配置'
            }), 400
        
        # 获取PSO参数
        pso_params = data.get('psoParameters', {})
        num_particles = pso_params.get('num_particles', 30)
        max_iter = pso_params.get('max_iter', 50)
        w = pso_params.get('w', 0.8)
        c1 = pso_params.get('c1', 1.5)
        c2 = pso_params.get('c2', 1.5)
        v_min = pso_params.get('v_min', 0.95)
        v_max = pso_params.get('v_max', 1.05)
        SB = pso_params.get('SB', 10)
        UB = pso_params.get('UB', 10.38)
        pr = pso_params.get('pr', 1e-6)
        
        # 转换数据格式
        bus_data = np.array(data['busData'])
        branch_data = np.array(data['branchData'])
        tunable_nodes = data['tunableNodes']
        
        # 执行PSO优化
        result = pso_op(
            Bus=bus_data,
            Branch=branch_data,
            tunable_q_nodes=tunable_nodes,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
            v_min=v_min,
            v_max=v_max,
            SB=SB,
            UB=UB,
            pr=pr
        )
        
        # 保存优化结果
        result_file = os.path.join(RESULTS_DIR, f'pso_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'PSO优化失败'
        }), 500

@pso_optimize_bp.route('/api/pso-optimize/optimize/v2', methods=['POST'])
def pso_optimize_v2():
    """执行PSO无功优化（v2算法）"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'busData' not in data or 'branchData' not in data or 'tunableNodes' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需的节点数据、支路数据或可调节点配置'
            }), 400
        
        # 获取PSO参数
        pso_params = data.get('psoParameters', {})
        num_particles = pso_params.get('num_particles', 30)
        max_iter = pso_params.get('max_iter', 50)
        w = pso_params.get('w', 0.8)
        c1 = pso_params.get('c1', 1.5)
        c2 = pso_params.get('c2', 1.5)
        v_min = pso_params.get('v_min', 0.95)
        v_max = pso_params.get('v_max', 1.05)
        SB = pso_params.get('SB', 10)
        UB = pso_params.get('UB', 10.38)
        pr = pso_params.get('pr', 1e-6)
        
        # 转换数据格式
        bus_data = np.array(data['busData'])
        branch_data = np.array(data['branchData'])
        tunable_nodes = data['tunableNodes']
        
        # 执行PSO优化（v2）
        result = pso_op_v2(
            Bus=bus_data,
            Branch=branch_data,
            tunable_q_nodes=tunable_nodes,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
            v_min=v_min,
            v_max=v_max,
            SB=SB,
            UB=UB,
            pr=pr
        )
        
        # 保存优化结果
        result_file = os.path.join(RESULTS_DIR, f'pso_result_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'PSO优化失败(v2)'
        }), 500

@pso_optimize_bp.route('/api/pso-optimize/template', methods=['GET'])
def get_pso_template():
    """获取PSO优化示例数据模板"""
    try:
        template = {
            'busData': [
                [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
                [6, 0, 0], [7, 0, 0], [8, 0, 0], [9, 0, 0], [10, 0, 0],
                [11, 0.325071, -0.023704], [12, 0, 0], [13, -0.02834, 0],
                [14, 0.20565, 0.00301], [15, 0, 0], [16, -0.067145, 0],
                [17, 0.095275, 0.00102], [18, 0, 0], [19, -0.09425, 0],
                [20, 0.24197, 0.02307], [21, 0, 0], [22, -0.052715, 0],
                [23, 0.214965, 0.05159], [24, 0, 0], [25, -0.1508, 0],
                [26, 0.231172, 0.007432], [27, 0.121383, 0.02996], [28, 0, 0],
                [29, -0.47385, 0], [30, 0.648046, 0.163962], [31, 0, 0],
                [32, 0.01908, 0.00991], [33, -0.6704, 0.0386], [34, -1.205, 0.414]
            ],
            'branchData': [
                [1, 1, 2, 0.3436, 0.8136], [2, 2, 3, 0.0341, 0.0807],
                [3, 3, 4, 0.0369, 0.0874], [4, 4, 5, 0.0426, 0.1009],
                [5, 5, 6, 0.0852, 0.2017], [6, 6, 7, 0.4885, 1.1565],
                [7, 7, 8, 0.7128, 1.6877], [8, 8, 9, 0.3280, 0.7766],
                [9, 9, 10, 0.3124, 0.7396], [10, 10, 11, 0.4686, 1.1095],
                [11, 2, 12, 0.2456, 0.1778], [12, 12, 13, 0.0238, 0.0172],
                [13, 12, 14, 0.0238, 0.0172], [14, 3, 15, 0.0475, 0.0344],
                [15, 15, 16, 0.0238, 0.0172], [16, 15, 17, 0.0238, 0.0172],
                [17, 4, 18, 0.4275, 0.3096], [18, 18, 19, 0.0238, 0.0172],
                [19, 18, 20, 0.0238, 0.0172], [20, 5, 21, 0.2233, 0.1617],
                [21, 21, 22, 0.0238, 0.0172], [22, 21, 23, 0.0238, 0.0172],
                [23, 6, 24, 0.4275, 0.3096], [24, 24, 25, 0.02375, 0.0172],
                [25, 24, 26, 0.02375, 0.0172], [26, 7, 27, 0.35625, 0.258],
                [27, 8, 28, 0.4061, 1.0168], [28, 28, 29, 0.02375, 0.0172],
                [29, 28, 30, 0.02375, 0.0172], [30, 9, 31, 0.475, 0.344],
                [31, 31, 32, 0.02375, 0.0172], [32, 31, 33, 0.02375, 0.0172],
                [33, 10, 34, 0.16625, 0.1204]
            ],
            'tunableNodes': [
                [32, -0.3, 0.3, "节点33"],
                [33, -0.5, 0.5, "节点34"]
            ],
            'psoParameters': {
                'num_particles': 30, 'max_iter': 50, 'w': 0.8,
                'c1': 1.5, 'c2': 1.5, 'v_min': 0.95, 'v_max': 1.05,
                'SB': 10, 'UB': 10.38, 'pr': 1e-6
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': template
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取PSO模板失败: {str(e)}'
        }), 500

@pso_optimize_bp.route('/api/pso-optimize/results', methods=['GET'])
def get_pso_results():
    """获取PSO优化结果列表"""
    try:
        results = []
        if os.path.exists(RESULTS_DIR):
            for filename in os.listdir(RESULTS_DIR):
                if filename.startswith('pso_result_') and filename.endswith('.json'):
                    filepath = os.path.join(RESULTS_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        results.append({
                            'filename': filename,
                            'timestamp': filename.replace('pso_result_', '').replace('.json', ''),
                            'optimal_loss_rate': result_data.get('optimal_loss_rate'),
                            'initial_loss_rate': result_data.get('initial_loss_rate'),
                            'loss_reduction_percent': result_data.get('loss_reduction_percent'),
                            'convergence': result_data.get('convergence')
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
            'message': f'获取PSO结果列表失败: {str(e)}'
        }), 500
