"""
测试样本管理模块
用于管理 TRAINING_DATA_DIR/{line_name}/test 目录下的 Excel 文件
"""
import os
import pandas as pd
from flask import jsonify


def get_test_samples_list(line_name, training_data_dir):
    """
    获取指定线路的测试样本列表

    Args:
        line_name: 线路名称
        training_data_dir: 训练数据根目录

    Returns:
        Flask response
    """
    try:
        test_dir = os.path.join(training_data_dir, line_name, 'test')

        if not os.path.exists(test_dir):
            return jsonify({
                'status': 'success',
                'data': []
            })

        samples = []
        for filename in os.listdir(test_dir):
            if filename.endswith(('.xlsx', '.xls')):
                filepath = os.path.join(test_dir, filename)
                file_stat = os.stat(filepath)

                samples.append({
                    'filename': filename,
                    'name': os.path.splitext(filename)[0],
                    'size': file_stat.st_size,
                    'modified_time': file_stat.st_mtime
                })

        # 按修改时间排序
        samples.sort(key=lambda x: x.get('modified_time', 0), reverse=True)

        return jsonify({
            'status': 'success',
            'data': samples
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'获取测试样本列表失败: {str(e)}'
        }), 500


def get_test_sample_detail(line_name, filename, training_data_dir):
    """
    获取测试样本详情（读取 Excel 文件的 3 个 sheet: date, slack, bus）

    Args:
        line_name: 线路名称
        filename: Excel 文件名
        training_data_dir: 训练数据根目录

    Returns:
        Flask response - 返回结构化的历史工况数据
    """
    try:
        test_dir = os.path.join(training_data_dir, line_name, 'test')
        filepath = os.path.join(test_dir, filename)

        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': '测试样本文件不存在'
            }), 404

        # 读取 Excel 文件的 3 个 sheet
        excel_file = pd.ExcelFile(filepath)

        # 检查必需的 sheet 是否存在
        required_sheets = ['date', 'slack', 'bus']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
        if missing_sheets:
            return jsonify({
                'status': 'error',
                'message': f'Excel 文件缺少必需的 sheet: {", ".join(missing_sheets)}'
            }), 400

        # 读取 date sheet（断面时间）- 跳过第一行标题
        date_df = pd.read_excel(filepath, sheet_name='date', header=None, skiprows=1)
        history_oc_date = str(date_df.iloc[0, 0]) if not date_df.empty else ''

        # 读取 slack sheet（母线电压）- 跳过第一行标题
        slack_df = pd.read_excel(filepath, sheet_name='slack', header=None, skiprows=1)
        bus_voltage = str(slack_df.iloc[0, 0]) if not slack_df.empty else ''

        # 读取 bus sheet（节点负荷数据）- 跳过第一行标题
        bus_df = pd.read_excel(filepath, sheet_name='bus', header=None, skiprows=1)

        # 转换 bus 数据为数组格式 [节点号, 有功负荷, 无功负荷]
        history_oc_data = []
        for _, row in bus_df.iterrows():
            history_oc_data.append([
                str(row[0]),  # 节点号
                str(row[1]),  # 有功负荷
                str(row[2])   # 无功负荷
            ])

        # 构造返回数据
        data = {
            'filename': filename,
            'historyOcDate': history_oc_date,
            'busVoltage': bus_voltage,
            'historyOcData': history_oc_data
        }

        return jsonify({
            'status': 'success',
            'data': data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'获取测试样本详情失败: {str(e)}'
        }), 500
