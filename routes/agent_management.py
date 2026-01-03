"""
智能体管理模块
"""
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import jsonify


def get_timezone():
    """获取时区配置，默认为 Asia/Shanghai"""
    tz_name = os.environ.get('TZ', 'Asia/Shanghai')
    return ZoneInfo(tz_name)


def get_agents_list(line_id):
    """
    获取指定线路的历史训练智能体（模型）列表
    包含模型的详细信息和训练历史

    Args:
        line_id: 线路ID

    Returns:
        Flask response
    """
    try:
        agents = []

        # 扫描指定线路的agent目录
        agent_dir = os.path.join('data', 'line_data', line_id, 'agent')
        if os.path.isdir(agent_dir):
            for filename in os.listdir(agent_dir):
                if filename.endswith('.npz'):
                    filepath = os.path.join(agent_dir, filename)
                    file_stat = os.stat(filepath)

                    # 尝试读取对应的训练历史文件
                    history_file = filepath.replace('.npz', '_history.json')
                    training_history = None
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'r', encoding='utf-8') as f:
                                training_history = json.load(f)
                        except:
                            pass

                    # 使用环境变量配置的时区
                    tz = get_timezone()
                    created_time = datetime.fromtimestamp(file_stat.st_ctime, tz=tz).strftime('%Y-%m-%d %H:%M:%S')
                    modified_time = datetime.fromtimestamp(file_stat.st_mtime, tz=tz).strftime('%Y-%m-%d %H:%M:%S')
                    
                    agents.append({
                        'filename': filename,
                        'model_name': filename.replace('.npz', ''),
                        'line_id': line_id,
                        'size': file_stat.st_size,
                        'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                        'created_time': created_time,
                        'modified_time': modified_time,
                        'training_history': training_history,
                        'location': f'line_data/{line_id}/agent'
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


def delete_agent_by_name(model_name, line_id):
    """
    删除智能体（模型）

    Args:
        model_name: 模型名称
        line_id: 线路ID

    Returns:
        Flask response
    """
    try:
        # 查找并删除模型文件
        deleted = False
        agent_dir = os.path.join('data', 'line_data', line_id, 'agent')
        model_file = os.path.join(agent_dir, f'{model_name}.npz')

        if os.path.exists(model_file):
            os.remove(model_file)
            deleted = True

            # 删除相关文件
            for ext in ['_history.json', '_meta.json']:
                related_file = model_file.replace('.npz', ext)
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
