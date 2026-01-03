"""
线路管理服务模块
负责线路的增删改查以及相关文件管理
"""
import os
import json
import uuid
import shutil
import zipfile
import tarfile
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any


class LineService:
    """线路管理服务类"""

    def __init__(self, base_dir: str = None):
        """
        初始化线路服务

        Args:
            base_dir: 线路数据基础目录，默认为 data/line_data
        """
        if base_dir is None:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(current_dir, 'data', 'line_data')

        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_line_dir(self, line_id: str) -> str:
        """获取线路目录路径"""
        return os.path.join(self.base_dir, line_id)

    def _get_line_json_path(self, line_id: str) -> str:
        """获取线路JSON文件路径"""
        return os.path.join(self._get_line_dir(line_id), 'line_info.json')

    def _read_line_json(self, line_id: str) -> Optional[Dict[str, Any]]:
        """读取线路JSON数据"""
        json_path = self._get_line_json_path(line_id)
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] 读取线路JSON失败: {e}")
            return None

    def _write_line_json(self, line_id: str, data: Dict[str, Any]) -> bool:
        """写入线路JSON数据"""
        json_path = self._get_line_json_path(line_id)

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            return True
        except Exception as e:
            print(f"[ERROR] 写入线路JSON失败: {e}")
            return False

    def _extract_archive(self, archive_path: str, extract_dir: str) -> bool:
        """
        解压压缩包，自动去掉外层目录

        Args:
            archive_path: 压缩包路径
            extract_dir: 解压目标目录

        Returns:
            是否成功
        """
        try:
            os.makedirs(extract_dir, exist_ok=True)
            
            # 创建临时解压目录
            temp_extract_dir = os.path.join(extract_dir, '_temp_extract')
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            # 解压到临时目录
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
            elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_extract_dir)
            else:
                shutil.rmtree(temp_extract_dir)
                return False
            
            # 检查解压后的内容
            items = os.listdir(temp_extract_dir)
            
            # 如果只有一个目录，则认为是外层包装目录，需要去掉
            if len(items) == 1:
                single_item = os.path.join(temp_extract_dir, items[0])
                if os.path.isdir(single_item):
                    # 将内层目录的内容移动到目标目录
                    for item in os.listdir(single_item):
                        src = os.path.join(single_item, item)
                        dst = os.path.join(extract_dir, item)
                        shutil.move(src, dst)
                else:
                    # 如果是单个文件，直接移动
                    dst = os.path.join(extract_dir, items[0])
                    shutil.move(single_item, dst)
            else:
                # 如果有多个文件/目录，直接移动所有内容
                for item in items:
                    src = os.path.join(temp_extract_dir, item)
                    dst = os.path.join(extract_dir, item)
                    shutil.move(src, dst)
            
            # 清理临时目录
            shutil.rmtree(temp_extract_dir)
            
            return True
        except Exception as e:
            print(f"[ERROR] 解压文件失败: {e}")
            # 清理临时目录
            temp_extract_dir = os.path.join(extract_dir, '_temp_extract')
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
            return False

    def create_line(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        创建新线路

        Args:
            data: 线路数据，包含：
                - name: 线路名称
                - ring: 所在标准环
                - configStatus: 工况配置 ('1' 或 '2')
                - status: 状态 ('0' 或 '1')
                - 其他可选字段

        Returns:
            创建的线路数据（包含id），失败返回None
        """
        try:
            # 生成UUID
            line_id = str(uuid.uuid4())
            line_dir = self._get_line_dir(line_id)
            
            # 创建线路目录
            os.makedirs(line_dir, exist_ok=True)
            
            # 创建train和test子目录
            os.makedirs(os.path.join(line_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(line_dir, 'test'), exist_ok=True)
            
            # 准备线路数据
            now = datetime.now().isoformat()
            line_data = {
                'id': line_id,
                'name': data.get('name'),
                'ring': data.get('ring'),
                'configStatus': data.get('configStatus', '1'),
                'status': data.get('status', '1'),
                'createdAt': now,
                'updatedAt': now,
            }
            
            # 保存可选字段
            optional_fields = [
                'lineParams', 'historyOc', 'adjustablePv',
                'voltageLimit', 'keyNodes', 'historyOcDate',
                'busVoltage', 'trainingSamplePath', 'testSamplePath',
                'lineParamsData', 'historyOcData', 'adjustablePvData',
                'voltageLimitData', 'keyNodesData'
            ]
            for field in optional_fields:
                if field in data:
                    line_data[field] = data[field]
            
            # 写入JSON文件
            if not self._write_line_json(line_id, line_data):
                shutil.rmtree(line_dir)
                return None
            
            return line_data
            
        except Exception as e:
            print(f"[ERROR] 创建线路失败: {e}")
            if os.path.exists(line_dir):
                shutil.rmtree(line_dir)
            return None

    def get_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        """
        获取线路信息

        Args:
            line_id: 线路ID

        Returns:
            线路数据，不存在返回None
        """
        return self._read_line_json(line_id)

    def get_all_lines(self) -> List[Dict[str, Any]]:
        """
        获取所有线路列表

        Returns:
            线路列表
        """
        lines = []
        
        try:
            if not os.path.exists(self.base_dir):
                return lines
            
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    line_data = self._read_line_json(item)
                    if line_data:
                        lines.append(line_data)
            
            # 按创建时间排序
            lines.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
            
        except Exception as e:
            print(f"[ERROR] 获取线路列表失败: {e}")
        
        return lines

    def update_line(self, line_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        更新线路信息

        Args:
            line_id: 线路ID
            data: 要更新的数据

        Returns:
            更新后的线路数据，失败返回None
        """
        try:
            # 读取现有数据
            line_data = self._read_line_json(line_id)
            if not line_data:
                return None
            
            # 更新字段
            for key, value in data.items():
                if key not in ['id', 'createdAt']:  # 不允许修改id和创建时间
                    line_data[key] = value
            
            # 更新修改时间
            line_data['updatedAt'] = datetime.now().isoformat()
            
            # 写入JSON
            if not self._write_line_json(line_id, line_data):
                return None
            
            return line_data
            
        except Exception as e:
            print(f"[ERROR] 更新线路失败: {e}")
            return None

    def delete_line(self, line_id: str) -> bool:
        """
        删除线路

        Args:
            line_id: 线路ID

        Returns:
            是否成功
        """
        try:
            line_dir = self._get_line_dir(line_id)
            
            if not os.path.exists(line_dir):
                return False
            
            shutil.rmtree(line_dir)
            return True
            
        except Exception as e:
            print(f"[ERROR] 删除线路失败: {e}")
            return False

    def upload_training_sample(self, line_id: str, archive_path: str) -> bool:
        """
        上传训练样本压缩包

        Args:
            line_id: 线路ID
            archive_path: 压缩包路径

        Returns:
            是否成功
        """
        try:
            line_dir = self._get_line_dir(line_id)
            if not os.path.exists(line_dir):
                return False
            
            train_dir = os.path.join(line_dir, 'train')
            
            # 清空train目录
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir, exist_ok=True)
            
            # 解压到train目录
            if not self._extract_archive(archive_path, train_dir):
                return False
            
            # 更新线路信息
            line_data = self._read_line_json(line_id)
            if line_data:
                line_data['trainingSamplePath'] = 'train'
                line_data['updatedAt'] = datetime.now().isoformat()
                self._write_line_json(line_id, line_data)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 上传训练样本失败: {e}")
            return False

    def upload_test_sample(self, line_id: str, archive_path: str) -> bool:
        """
        上传测试样本压缩包

        Args:
            line_id: 线路ID
            archive_path: 压缩包路径

        Returns:
            是否成功
        """
        try:
            line_dir = self._get_line_dir(line_id)
            if not os.path.exists(line_dir):
                return False
            
            test_dir = os.path.join(line_dir, 'test')
            
            # 清空test目录
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            # 解压到test目录
            if not self._extract_archive(archive_path, test_dir):
                return False
            
            # 更新线路信息
            line_data = self._read_line_json(line_id)
            if line_data:
                line_data['testSamplePath'] = 'test'
                line_data['updatedAt'] = datetime.now().isoformat()
                self._write_line_json(line_id, line_data)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 上传测试样本失败: {e}")
            return False

    def get_training_samples(self, line_id: str) -> List[Dict[str, Any]]:
        """
        获取训练样本列表

        Args:
            line_id: 线路ID

        Returns:
            训练样本列表
        """
        try:
            line_dir = self._get_line_dir(line_id)
            train_dir = os.path.join(line_dir, 'train')

            if not os.path.exists(train_dir):
                return []

            samples = []
            for filename in os.listdir(train_dir):
                if filename.endswith(('.xlsx', '.xls')):
                    filepath = os.path.join(train_dir, filename)
                    file_stat = os.stat(filepath)

                    samples.append({
                        'filename': filename,
                        'name': os.path.splitext(filename)[0],
                        'size': file_stat.st_size,
                        'modifiedTime': file_stat.st_mtime
                    })

            samples.sort(key=lambda x: x.get('modifiedTime', 0), reverse=True)
            return samples

        except Exception as e:
            print(f"[ERROR] 获取训练样本列表失败: {e}")
            return []

    def get_test_samples(self, line_id: str) -> List[Dict[str, Any]]:
        """
        获取测试样本列表

        Args:
            line_id: 线路ID

        Returns:
            测试样本列表
        """
        try:
            line_dir = self._get_line_dir(line_id)
            test_dir = os.path.join(line_dir, 'test')

            if not os.path.exists(test_dir):
                return []

            samples = []
            for filename in os.listdir(test_dir):
                if filename.endswith(('.xlsx', '.xls')):
                    filepath = os.path.join(test_dir, filename)
                    file_stat = os.stat(filepath)

                    samples.append({
                        'filename': filename,
                        'name': os.path.splitext(filename)[0],
                        'size': file_stat.st_size,
                        'modifiedTime': file_stat.st_mtime
                    })

            samples.sort(key=lambda x: x.get('modifiedTime', 0), reverse=True)
            return samples

        except Exception as e:
            print(f"[ERROR] 获取测试样本列表失败: {e}")
            return []

    def get_training_sample_detail(self, line_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        获取训练样本详情

        Args:
            line_id: 线路ID
            filename: Excel 文件名

        Returns:
            样本详情数据，失败返回None
        """
        try:
            line_dir = self._get_line_dir(line_id)
            train_dir = os.path.join(line_dir, 'train')
            filepath = os.path.join(train_dir, filename)

            if not os.path.exists(filepath):
                return None

            excel_file = pd.ExcelFile(filepath)

            required_sheets = ['date', 'slack', 'bus']
            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                print(f"[ERROR] Excel 文件缺少必需的 sheet: {', '.join(missing_sheets)}")
                return None

            date_df = pd.read_excel(filepath, sheet_name='date', header=None, skiprows=1)
            history_oc_date = str(date_df.iloc[0, 0]) if not date_df.empty else ''

            slack_df = pd.read_excel(filepath, sheet_name='slack', header=None, skiprows=1)
            bus_voltage = str(slack_df.iloc[0, 0]) if not slack_df.empty else ''

            bus_df = pd.read_excel(filepath, sheet_name='bus', header=None, skiprows=1)

            history_oc_data = []
            for _, row in bus_df.iterrows():
                history_oc_data.append([
                    str(row[0]),
                    str(row[1]),
                    str(row[2])
                ])

            return {
                'filename': filename,
                'historyOcDate': history_oc_date,
                'busVoltage': bus_voltage,
                'historyOcData': history_oc_data
            }

        except Exception as e:
            print(f"[ERROR] 获取训练样本详情失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def get_test_sample_detail(self, line_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        获取测试样本详情

        Args:
            line_id: 线路ID
            filename: Excel 文件名

        Returns:
            样本详情数据，失败返回None
        """
        try:
            line_dir = self._get_line_dir(line_id)
            test_dir = os.path.join(line_dir, 'test')
            filepath = os.path.join(test_dir, filename)

            if not os.path.exists(filepath):
                return None

            excel_file = pd.ExcelFile(filepath)

            required_sheets = ['date', 'slack', 'bus']
            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                print(f"[ERROR] Excel 文件缺少必需的 sheet: {', '.join(missing_sheets)}")
                return None

            date_df = pd.read_excel(filepath, sheet_name='date', header=None, skiprows=1)
            history_oc_date = str(date_df.iloc[0, 0]) if not date_df.empty else ''

            slack_df = pd.read_excel(filepath, sheet_name='slack', header=None, skiprows=1)
            bus_voltage = str(slack_df.iloc[0, 0]) if not slack_df.empty else ''

            bus_df = pd.read_excel(filepath, sheet_name='bus', header=None, skiprows=1)

            history_oc_data = []
            for _, row in bus_df.iterrows():
                history_oc_data.append([
                    str(row[0]),
                    str(row[1]),
                    str(row[2])
                ])

            return {
                'filename': filename,
                'historyOcDate': history_oc_date,
                'busVoltage': bus_voltage,
                'historyOcData': history_oc_data
            }

        except Exception as e:
            print(f"[ERROR] 获取测试样本详情失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None


# 全局线路服务实例
line_service = LineService()
