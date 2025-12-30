"""
潮流计算相关路由
"""
from flask import Blueprint, jsonify, request, Response
import json
import numpy as np
import os
from datetime import datetime
from lib.powerflow_json import power_flow_calculation
from lib.oc_source import get_oc_source_flat_data
from lib.oc_source_remote import get_remote_oc_source_data

flow_compute_bp = Blueprint('flow_compute', __name__)

# 配置目录
DATA_DIR = 'data'
RESULTS_DIR = 'results'

@flow_compute_bp.route('/api/flow-compute/parameters', methods=['POST'])
def save_parameters():
    """保存计算参数"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_fields = ['baseVoltage', 'basePower', 'convergencePrecision']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必需参数: {field}'
                }), 400
        
        # 保存参数到文件
        params_file = os.path.join(DATA_DIR, 'parameters.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': '参数保存成功',
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'参数保存失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/parameters', methods=['GET'])
def get_parameters():
    """获取保存的计算参数"""
    try:
        params_file = os.path.join(DATA_DIR, 'parameters.json')
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
                'baseVoltage': '10.3',
                'basePower': '10.38',
                'convergencePrecision': '1e-6'
            }
            return jsonify({
                'status': 'success',
                'data': default_params
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取参数失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/calculate', methods=['POST'])
def calculate_power_flow():
    """执行潮流计算"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'busData' not in data or 'branchData' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需的节点数据或支路数据'
            }), 400
        
        # 获取计算参数
        base_voltage = float(data.get('baseVoltage', 10.38))
        base_power = float(data.get('basePower', 10))
        precision = float(data.get('convergencePrecision', 1e-6))
        
        # 转换数据格式
        bus_data = np.array(data['busData'])
        branch_data = np.array(data['branchData'])
        
        # 执行潮流计算
        result = power_flow_calculation(
            bus_data, 
            branch_data, 
            SB=base_power, 
            UB=base_voltage, 
            pr=precision
        )
        
        # 解析计算结果
        result_data = json.loads(result)
        
        # 保存计算结果
        result_file = os.path.join(RESULTS_DIR, f'result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        if result_data['status'] == 'error':
            return jsonify({
                'status': 'error',
                'message': '服务端错误，请联系管理员'
            }), 500
        
        del result_data['status'];
        return jsonify({
            'status': 'success',
            'data': result_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'潮流计算失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/template', methods=['GET'])
def get_template():
    """获取示例数据模板"""
    try:
        # 从powerflow_json.py中的示例数据创建模板
        template = {
            'busData': [
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [8, 0, 0],
                [9, 0, 0],
                [10, 0, 0],
                [11, 0.325071, -0.023704],
                [12, 0, 0],
                [13, -0.02834, 0],
                [14, 0.20565, 0.00301],
                [15, 0, 0],
                [16, -0.067145, 0],
                [17, 0.095275, 0.00102],
                [18, 0, 0],
                [19, -0.09425, 0],
                [20, 0.24197, 0.02307],
                [21, 0, 0],
                [22, -0.052715, 0],
                [23, 0.214965, 0.05159],
                [24, 0, 0],
                [25, -0.1508, 0],
                [26, 0.231172, 0.007432],
                [27, 0.121383, 0.02996],
                [28, 0, 0],
                [29, -0.47385, 0],
                [30, 0.648046, 0.163962],
                [31, 0, 0],
                [32, 0.01908, 0.00991],
                [33, -0.6704, -0.1258],
                [34, -1.205, -0.1140]
            ],
            'branchData': [
                [1, 1, 2, 0.3436, 0.8136],
                [2, 2, 3, 0.0341, 0.0807],
                [3, 3, 4, 0.0369, 0.0874],
                [4, 4, 5, 0.0426, 0.1009],
                [5, 5, 6, 0.0852, 0.2017],
                [6, 6, 7, 0.4885, 1.1565],
                [7, 7, 8, 0.7128, 1.6877],
                [8, 8, 9, 0.3280, 0.7766],
                [9, 9, 10, 0.3124, 0.7396],
                [10, 10, 11, 0.4686, 1.1095],
                [11, 2, 12, 0.2456, 0.1778],
                [12, 12, 13, 0.0238, 0.0172],
                [13, 12, 14, 0.0238, 0.0172],
                [14, 3, 15, 0.0475, 0.0344],
                [15, 15, 16, 0.0238, 0.0172],
                [16, 15, 17, 0.0238, 0.0172],
                [17, 4, 18, 0.4275, 0.3096],
                [18, 18, 19, 0.0238, 0.0172],
                [19, 18, 20, 0.0238, 0.0172],
                [20, 5, 21, 0.2233, 0.1617],
                [21, 21, 22, 0.0238, 0.0172],
                [22, 21, 23, 0.0238, 0.0172],
                [23, 6, 24, 0.4275, 0.3096],
                [24, 24, 25, 0.02375, 0.0172],
                [25, 24, 26, 0.02375, 0.0172],
                [26, 7, 27, 0.35625, 0.258],
                [27, 8, 28, 0.4061, 1.0168],
                [28, 28, 29, 0.02375, 0.0172],
                [29, 28, 30, 0.02375, 0.0172],
                [30, 9, 31, 0.475, 0.344],
                [31, 31, 32, 0.02375, 0.0172],
                [32, 31, 33, 0.02375, 0.0172],
                [33, 10, 34, 0.16625, 0.1204]
            ],
            'parameters': {
                'baseVoltage': 10.3,
                'basePower': 10.38,
                'convergencePrecision': 1e-6
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': template
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取模板失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/results', methods=['GET'])
def get_results():
    """获取历史计算结果列表"""
    try:
        results = []
        if os.path.exists(RESULTS_DIR):
            for filename in os.listdir(RESULTS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(RESULTS_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        results.append({
                            'filename': filename,
                            'timestamp': filename.replace('result_', '').replace('.json', ''),
                            'status': result_data.get('status', 'unknown'),
                            'summary': result_data.get('summary', {})
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
            'message': f'获取结果列表失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/results/<filename>', methods=['GET'])
def get_result_detail(filename):
    """获取具体计算结果详情"""
    try:
        result_file = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(result_file):
            return jsonify({
                'status': 'error',
                'message': '结果文件不存在'
            }), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        return jsonify(result_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取结果详情失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/export/<filename>', methods=['GET'])
def export_result(filename):
    """导出计算结果"""
    try:
        result_file = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(result_file):
            return jsonify({
                'status': 'error',
                'message': '结果文件不存在'
            }), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 返回JSON格式的下载响应
        return Response(
            json.dumps(result_data, ensure_ascii=False, indent=2),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'导出结果失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/oc-source', methods=['GET'])
def get_remote_oc_source():
    """获取实时运行工况数据源（从宿主机接口）"""
    try:
        flat_data, error_message = get_remote_oc_source_data()
        
        if error_message:
            # 根据错误类型返回不同的HTTP状态码
            if '无法连接' in error_message or '超时' in error_message:
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), 503
            elif '未找到' in error_message or '未能成功配对' in error_message:
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), 404
            else:
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), 400
        
        return jsonify({
            'status': 'success',
            'data': flat_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取数据源失败: {str(e)}'
        }), 500

@flow_compute_bp.route('/api/flow-compute/oc-source-local', methods=['GET'])
def get_oc_source():
    """获取实时运行工况数据源（从本地JSON文件）"""
    try:
        flat_data, error_message = get_oc_source_flat_data()
        
        if error_message:
            # 根据错误类型返回不同的HTTP状态码
            if '目录不存在' in error_message or '文件不存在' in error_message:
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), 404
            else:
                return jsonify({
                    'status': 'error',
                    'message': error_message
                }), 400
        
        return jsonify({
            'status': 'success',
            'data': flat_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取数据源失败: {str(e)}'
        }), 500
