"""
训练数据上传相关功能模块
"""
import os
import shutil
import zipfile
import tarfile
import threading
from datetime import datetime
from flask import jsonify, request
from werkzeug.utils import secure_filename
from .upload_task_manager import upload_task_manager


def validate_powerdata_structure(extract_path):
    """
    验证解压后的目录结构是否符合要求
    支持两种结构：
    1. 根目录直接包含线路目录（C5336, C5347等）
    2. 根目录包含一个任意名称的文件夹，该文件夹下包含线路目录

    每条线路下固定包含：hisdata, modeldata, realdata, test, train 五个目录
    hisdata下固定包含 alldata, psopvdata, pvdata 三个目录

    返回: (is_valid, error_message, line_dirs, actual_root_path)
    """
    try:
        required_dirs = ['hisdata', 'modeldata', 'realdata', 'test', 'train']
        required_hisdata_subdirs = ['alldata', 'psopvdata', 'pvdata']

        def is_line_directory(path, dir_name):
            """检查是否是有效的线路目录，返回(is_valid, missing_info)"""
            if not os.path.isdir(path):
                return False, f"{dir_name}: 不是目录"

            missing_dirs = []
            # 检查是否包含必需的5个目录
            for req_dir in required_dirs:
                if not os.path.exists(os.path.join(path, req_dir)):
                    missing_dirs.append(req_dir)

            if missing_dirs:
                return False, f"{dir_name}: 缺少目录 {', '.join(missing_dirs)}"

            # 检查hisdata下的3个子目录
            hisdata_path = os.path.join(path, 'hisdata')
            missing_hisdata = []
            for subdir in required_hisdata_subdirs:
                if not os.path.exists(os.path.join(hisdata_path, subdir)):
                    missing_hisdata.append(subdir)

            if missing_hisdata:
                return False, f"{dir_name}/hisdata: 缺少子目录 {', '.join(missing_hisdata)}"

            return True, ""

        # 获取根目录下的所有项，过滤掉系统目录
        items = os.listdir(extract_path)
        dirs = [d for d in items if os.path.isdir(os.path.join(extract_path, d))
                and not d.startswith('__MACOSX') and not d.startswith('.')]

        if not dirs:
            return False, "压缩包为空或不包含任何目录", [], extract_path

        print(f"[DEBUG] 根目录下的目录（已过滤系统目录）: {dirs}")

        # 情况1: 检查根目录下是否直接包含线路目录
        line_dirs = []
        validation_info = []
        for d in dirs:
            dir_path = os.path.join(extract_path, d)
            is_valid, info = is_line_directory(dir_path, d)
            if is_valid:
                line_dirs.append(d)
            else:
                validation_info.append(info)

        if line_dirs:
            # 找到了线路目录，直接返回
            print(f"[DEBUG] 在根目录找到线路目录: {line_dirs}")
            return True, "", line_dirs, extract_path

        # 情况2: 根目录下只有一个文件夹，检查该文件夹下是否包含线路目录
        if len(dirs) == 1:
            sub_root = os.path.join(extract_path, dirs[0])
            sub_items = os.listdir(sub_root)
            sub_dirs = [d for d in sub_items if os.path.isdir(os.path.join(sub_root, d))]

            print(f"[DEBUG] 子目录 {dirs[0]} 下的目录: {sub_dirs}")

            line_dirs = []
            sub_validation_info = []
            for d in sub_dirs:
                dir_path = os.path.join(sub_root, d)
                is_valid, info = is_line_directory(dir_path, d)
                if is_valid:
                    line_dirs.append(d)
                else:
                    sub_validation_info.append(info)

            if line_dirs:
                # 在子目录中找到了线路目录
                print(f"[DEBUG] 在子目录找到线路目录: {line_dirs}")
                return True, "", line_dirs, sub_root

            # 输出详细的验证失败信息
            error_details = "\n".join(sub_validation_info[:3])  # 只显示前3个
            return False, f"未找到符合要求的线路目录结构。\n检查结果:\n{error_details}", [], extract_path

        # 没有找到有效的线路目录
        error_details = "\n".join(validation_info[:3])  # 只显示前3个
        return False, f"未找到符合要求的线路目录结构。\n检查结果:\n{error_details}", [], extract_path

    except Exception as e:
        import traceback
        print(f"[ERROR] 验证目录结构异常: {traceback.format_exc()}")
        return False, f"验证目录结构时出错: {str(e)}", [], extract_path


def handle_upload_powerdata(training_data_dir):
    """
    处理训练数据压缩包上传（异步模式）
    文件上传完成后立即返回任务ID，后台处理解压和验证

    Args:
        training_data_dir: 训练数据保存目录

    Returns:
        Flask response with task_id
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '未找到上传文件'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '未选择文件'
            }), 400

        # 验证文件类型
        filename = secure_filename(file.filename)
        if not (filename.endswith('.zip') or filename.endswith('.tar') or
                filename.endswith('.tar.gz') or filename.endswith('.tgz')):
            return jsonify({
                'status': 'error',
                'message': '不支持的文件格式，仅支持 .zip, .tar, .tar.gz, .tgz'
            }), 400

        # 获取文件大小
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到文件开头

        # 创建任务
        task_id = upload_task_manager.create_task(filename, file_size)

        # 创建临时目录
        temp_dir = os.path.join(training_data_dir, f'temp_{task_id}')
        os.makedirs(temp_dir, exist_ok=True)

        # 保存上传的文件
        archive_path = os.path.join(temp_dir, filename)
        file.save(archive_path)

        # 更新任务状态
        upload_task_manager.update_task(task_id, status='processing', message='文件上传完成，开始处理...')

        # 启动后台线程处理
        thread = threading.Thread(
            target=upload_task_manager.process_upload,
            args=(task_id, archive_path, training_data_dir)
        )
        thread.daemon = True
        thread.start()

        # 立即返回任务ID
        return jsonify({
            'status': 'success',
            'message': '文件上传成功，正在后台处理',
            'data': {
                'task_id': task_id
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'上传失败: {str(e)}'
        }), 500


def get_upload_task_status(task_id):
    """
    获取上传任务状态

    Args:
        task_id: 任务ID

    Returns:
        Flask response
    """
    try:
        task = upload_task_manager.get_task(task_id)

        if not task:
            return jsonify({
                'status': 'error',
                'message': '任务不存在'
            }), 404

        return jsonify({
            'status': 'success',
            'data': task
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取任务状态失败: {str(e)}'
        }), 500


def get_powerdata_lines(training_data_dir):
    """
    获取已上传的线路数据列表

    Args:
        training_data_dir: 训练数据目录

    Returns:
        Flask response
    """
    try:
        lines = []
        if os.path.exists(training_data_dir):
            for item in os.listdir(training_data_dir):
                item_path = os.path.join(training_data_dir, item)
                if os.path.isdir(item_path) and not item.startswith('temp_'):
                    # 检查是否包含必需的子目录
                    required_dirs = ['hisdata', 'modeldata', 'realdata', 'test', 'train']
                    has_all_dirs = all(
                        os.path.exists(os.path.join(item_path, d))
                        for d in required_dirs
                    )

                    if has_all_dirs:
                        lines.append({
                            'name': item,
                            'path': item_path
                        })

        return jsonify({
            'status': 'success',
            'data': lines
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取线路列表失败: {str(e)}'
        }), 500

