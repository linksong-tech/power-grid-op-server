"""
样本管理模块
"""
import os
import json
from datetime import datetime
from flask import jsonify, request


def upload_samples(training_samples_dir):
    """
    上传训练样本

    Args:
        training_samples_dir: 训练样本目录

    Returns:
        Flask response
    """
    try:
        data = request.get_json()

        if 'samples' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需参数: samples'
            }), 400

        samples = data['samples']
        sample_name = data.get('name', f'samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # 保存样本到文件
        sample_file = os.path.join(training_samples_dir, f'{sample_name}.json')
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({
                'name': sample_name,
                'upload_time': datetime.now().isoformat(),
                'sample_count': len(samples),
                'samples': samples
            }, f, ensure_ascii=False, indent=2)

        return jsonify({
            'status': 'success',
            'message': '训练样本上传成功',
            'data': {
                'sample_name': sample_name,
                'sample_count': len(samples),
                'file_path': sample_file
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'上传训练样本失败: {str(e)}'
        }), 500


def get_samples_list(training_samples_dir):
    """
    获取已上传的训练样本列表

    Args:
        training_samples_dir: 训练样本目录

    Returns:
        Flask response
    """
    try:
        samples = []
        if os.path.exists(training_samples_dir):
            for filename in os.listdir(training_samples_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(training_samples_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        sample_data = json.load(f)
                        samples.append({
                            'filename': filename,
                            'name': sample_data.get('name', filename),
                            'upload_time': sample_data.get('upload_time', ''),
                            'sample_count': sample_data.get('sample_count', 0)
                        })

        # 按上传时间排序
        samples.sort(key=lambda x: x.get('upload_time', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'data': samples
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取训练样本列表失败: {str(e)}'
        }), 500


def get_sample_detail_by_filename(filename, training_samples_dir):
    """
    获取训练样本详情

    Args:
        filename: 样本文件名
        training_samples_dir: 训练样本目录

    Returns:
        Flask response
    """
    try:
        sample_file = os.path.join(training_samples_dir, filename)
        if not os.path.exists(sample_file):
            return jsonify({
                'status': 'error',
                'message': '样本文件不存在'
            }), 404

        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)

        return jsonify({
            'status': 'success',
            'data': sample_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取样本详情失败: {str(e)}'
        }), 500
