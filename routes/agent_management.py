"""
智能体管理模块
"""
import os
import json
from datetime import datetime
from flask import jsonify


def get_agents_list(models_dir, lib_dir):
    """
    获取历史训练的智能体（模型）列表
    包含模型的详细信息和训练历史

    Args:
        models_dir: 模型目录
        lib_dir: lib目录

    Returns:
        Flask response
    """
    try:
        agents = []

        # 扫描models目录
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pth'):
                    filepath = os.path.join(models_dir, filename)
                    file_stat = os.stat(filepath)

                    # 尝试读取对应的训练历史文件
                    history_file = filepath.replace('.pth', '_history.json')
                    training_history = None
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'r', encoding='utf-8') as f:
                                training_history = json.load(f)
                        except:
                            pass

                    agents.append({
                        'filename': filename,
                        'model_name': filename.replace('.pth', ''),
                        'size': file_stat.st_size,
                        'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                        'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'training_history': training_history,
                        'location': 'models'
                    })

        # 扫描lib目录
        if os.path.exists(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename.endswith('.pth'):
                    filepath = os.path.join(lib_dir, filename)
                    file_stat = os.stat(filepath)

                    agents.append({
                        'filename': filename,
                        'model_name': filename.replace('.pth', ''),
                        'size': file_stat.st_size,
                        'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                        'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'training_history': None,
                        'location': 'lib'
                    })

        # 按修改时间排序
        agents.sort(key=lambda x: x.get('modified_time', ''), reverse=True)

        return jsonify({
            'status': 'success',
            'data': agents
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取智能体列表失败: {str(e)}'
        }), 500


def delete_agent_by_name(model_name, models_dir):
    """
    删除智能体（模型）

    Args:
        model_name: 模型名称
        models_dir: 模型目录

    Returns:
        Flask response
    """
    try:
        # 查找并删除模型文件
        deleted = False
        model_file = os.path.join(models_dir, f'{model_name}.pth')
        if os.path.exists(model_file):
            os.remove(model_file)
            deleted = True

            # 删除相关文件
            for ext in ['_history.json', '_meta.json']:
                related_file = model_file.replace('.pth', ext)
                if os.path.exists(related_file):
                    os.remove(related_file)

        if not deleted:
            return jsonify({
                'status': 'error',
                'message': '模型文件不存在或无法删除'
            }), 404

        return jsonify({
            'status': 'success',
            'message': '智能体删除成功'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'删除智能体失败: {str(e)}'
        }), 500

