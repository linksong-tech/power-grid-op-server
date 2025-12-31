"""
上传任务管理模块
用于管理后台上传和解压任务
"""
import os
import json
import threading
import shutil
import zipfile
import tarfile
from datetime import datetime
from werkzeug.utils import secure_filename


class UploadTaskManager:
    """上传任务管理器"""

    def __init__(self):
        self.tasks = {}  # 任务字典: {task_id: task_info}
        self.lock = threading.Lock()

    def create_task(self, filename, file_size):
        """
        创建新任务

        Args:
            filename: 文件名
            file_size: 文件大小(字节)

        Returns:
            task_id: 任务ID
        """
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        with self.lock:
            self.tasks[task_id] = {
                'task_id': task_id,
                'filename': filename,
                'file_size': file_size,
                'status': 'uploading',  # uploading, processing, completed, failed
                'progress': 0,
                'message': '正在上传文件...',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'result': None,
                'error': None
            }

        return task_id

    def update_task(self, task_id, **kwargs):
        """
        更新任务状态

        Args:
            task_id: 任务ID
            **kwargs: 要更新的字段
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['updated_at'] = datetime.now().isoformat()

    def get_task(self, task_id):
        """
        获取任务信息

        Args:
            task_id: 任务ID

        Returns:
            task_info: 任务信息字典
        """
        with self.lock:
            return self.tasks.get(task_id)

    def process_upload(self, task_id, archive_path, training_data_dir):
        """
        后台处理上传的文件

        Args:
            task_id: 任务ID
            archive_path: 压缩包路径
            training_data_dir: 训练数据目录
        """
        temp_dir = None

        try:
            # 更新状态为处理中
            self.update_task(task_id, status='processing', progress=10, message='正在解压文件...')

            # 获取文件名
            filename = os.path.basename(archive_path)
            temp_dir = os.path.dirname(archive_path)

            # 解压文件
            extract_dir = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_dir, exist_ok=True)

            try:
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                elif filename.endswith(('.tar', '.tar.gz', '.tgz')):
                    with tarfile.open(archive_path, 'r:*') as tar_ref:
                        tar_ref.extractall(extract_dir)
            except Exception as e:
                raise Exception(f'解压文件失败: {str(e)}')

            self.update_task(task_id, progress=40, message='正在验证目录结构...')

            # 验证目录结构
            from .powerdata_upload import validate_powerdata_structure
            is_valid, error_msg, line_dirs, actual_root = validate_powerdata_structure(extract_dir)

            if not is_valid:
                raise Exception(f'目录结构验证失败: {error_msg}')

            self.update_task(task_id, progress=60, message='正在移动文件...')

            # 移动线路数据
            moved_lines = []
            for line_dir in line_dirs:
                src_path = os.path.join(actual_root, line_dir)
                dst_path = os.path.join(training_data_dir, line_dir)

                # 如果目标目录已存在，先删除
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)

                # 移动目录
                shutil.move(src_path, dst_path)
                moved_lines.append(line_dir)

            self.update_task(task_id, progress=90, message='正在清理临时文件...')

            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            # 任务完成
            result = {
                'lines': moved_lines,
                'line_count': len(moved_lines),
                'upload_time': datetime.now().isoformat()
            }

            self.update_task(
                task_id,
                status='completed',
                progress=100,
                message='上传完成',
                result=result
            )

        except Exception as e:
            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

            # 更新任务状态为失败
            self.update_task(
                task_id,
                status='failed',
                message='处理失败',
                error=str(e)
            )


# 全局任务管理器实例
upload_task_manager = UploadTaskManager()
