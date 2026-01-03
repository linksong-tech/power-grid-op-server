"""
线路管理路由
提供线路的增删改查接口
"""
import os
import tempfile
import shutil
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from lib.line_service import line_service

line_bp = Blueprint('line', __name__)


@line_bp.route('/api/lines', methods=['POST'])
def create_line():
    """创建新线路"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
        
        if not data.get('name'):
            return jsonify({'status': 'error', 'message': '线路名称不能为空'}), 400
        
        if not data.get('ring'):
            return jsonify({'status': 'error', 'message': '所在标准环不能为空'}), 400
        
        line_data = line_service.create_line(data)
        
        if not line_data:
            return jsonify({'status': 'error', 'message': '创建线路失败'}), 500
        
        return jsonify({'status': 'success', 'data': line_data})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'创建线路失败: {str(e)}'}), 500


@line_bp.route('/api/lines/all', methods=['GET'])
def get_all_lines_simple():
    """获取所有线路列表（简易版，不分页，仅返回id和name）"""
    try:
        all_lines = line_service.get_all_lines()

        # 只返回 id 和 name 字段
        simple_lines = [
            {'id': line['id'], 'name': line['name']}
            for line in all_lines
        ]

        return jsonify({
            'status': 'success',
            'data': simple_lines
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取线路列表失败: {str(e)}'}), 500


@line_bp.route('/api/lines', methods=['GET'])
def get_all_lines():
    """获取所有线路列表（支持分页）"""
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 10, type=int)

        # 获取所有线路
        all_lines = line_service.get_all_lines()
        total = len(all_lines)

        # 计算分页
        start = (page - 1) * page_size
        end = start + page_size
        lines = all_lines[start:end]

        return jsonify({
            'status': 'success',
            'data': {
                'list': lines,
                'total': total,
                'page': page,
                'pageSize': page_size
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取线路列表失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>', methods=['GET'])
def get_line(line_id):
    """获取单个线路信息"""
    try:
        line_data = line_service.get_line(line_id)
        
        if not line_data:
            return jsonify({'status': 'error', 'message': '线路不存在'}), 404
        
        return jsonify({'status': 'success', 'data': line_data})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取线路失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>', methods=['PUT'])
def update_line(line_id):
    """更新线路信息"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
        
        line_data = line_service.update_line(line_id, data)
        
        if not line_data:
            return jsonify({'status': 'error', 'message': '线路不存在或更新失败'}), 404
        
        return jsonify({'status': 'success', 'data': line_data})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'更新线路失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>', methods=['DELETE'])
def delete_line(line_id):
    """删除线路"""
    try:
        success = line_service.delete_line(line_id)
        
        if not success:
            return jsonify({'status': 'error', 'message': '线路不存在或删除失败'}), 404
        
        return jsonify({'status': 'success', 'message': '删除成功'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'删除线路失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/training-sample', methods=['POST'])
def upload_training_sample(line_id):
    """上传训练样本压缩包"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '未找到上传文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择文件'}), 400
        
        filename = secure_filename(file.filename)
        if not (filename.endswith('.zip') or filename.endswith('.tar') or
                filename.endswith('.tar.gz') or filename.endswith('.tgz')):
            return jsonify({'status': 'error', 'message': '不支持的文件格式'}), 400
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        success = line_service.upload_training_sample(line_id, temp_path)
        shutil.rmtree(temp_dir)
        
        if not success:
            return jsonify({'status': 'error', 'message': '上传训练样本失败'}), 500
        
        return jsonify({'status': 'success', 'message': '上传成功'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'上传失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/test-sample', methods=['POST'])
def upload_test_sample(line_id):
    """上传测试样本压缩包"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '未找到上传文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择文件'}), 400
        
        filename = secure_filename(file.filename)
        if not (filename.endswith('.zip') or filename.endswith('.tar') or
                filename.endswith('.tar.gz') or filename.endswith('.tgz')):
            return jsonify({'status': 'error', 'message': '不支持的文件格式'}), 400
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        success = line_service.upload_test_sample(line_id, temp_path)
        shutil.rmtree(temp_dir)
        
        if not success:
            return jsonify({'status': 'error', 'message': '上传测试样本失败'}), 500
        
        return jsonify({'status': 'success', 'message': '上传成功'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'上传失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/training-samples', methods=['GET'])
def get_training_samples(line_id):
    """获取训练样本列表"""
    try:
        samples = line_service.get_training_samples(line_id)
        return jsonify({'status': 'success', 'data': samples})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取训练样本列表失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/test-samples', methods=['GET'])
def get_test_samples(line_id):
    """获取测试样本列表"""
    try:
        samples = line_service.get_test_samples(line_id)
        return jsonify({'status': 'success', 'data': samples})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取测试样本列表失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/training-samples/<filename>', methods=['GET'])
def get_training_sample_detail(line_id, filename):
    """获取训练样本详情"""
    try:
        detail = line_service.get_training_sample_detail(line_id, filename)
        
        if not detail:
            return jsonify({'status': 'error', 'message': '训练样本不存在或读取失败'}), 404
        
        return jsonify({'status': 'success', 'data': detail})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取训练样本详情失败: {str(e)}'}), 500


@line_bp.route('/api/lines/<line_id>/test-samples/<filename>', methods=['GET'])
def get_test_sample_detail(line_id, filename):
    """获取测试样本详情"""
    try:
        detail = line_service.get_test_sample_detail(line_id, filename)
        
        if not detail:
            return jsonify({'status': 'error', 'message': '测试样本不存在或读取失败'}), 404
        
        return jsonify({'status': 'success', 'data': detail})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'获取测试样本详情失败: {str(e)}'}), 500
