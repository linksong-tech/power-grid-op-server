"""
TD3任务状态查询路由处理器
"""
from flask import jsonify
import copy

from routes.td3_config import batch_jobs, batch_jobs_lock
from .utils import format_datetime


def get_td3_batch_status(job_id: str):
    """获取批量评估任务状态（供前端短轮询）"""
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "任务不存在"}), 404

        # 深拷贝job数据以避免修改原始数据
        job_data = copy.deepcopy(job)

        # 格式化 current_sample_time 字段
        if "current_sample_time" in job_data:
            job_data["current_sample_time"] = format_datetime(job_data["current_sample_time"])

        # 格式化 result.results 中的 time 字段
        if "result" in job_data and isinstance(job_data["result"], dict):
            if "results" in job_data["result"] and isinstance(job_data["result"]["results"], list):
                for result_item in job_data["result"]["results"]:
                    if isinstance(result_item, dict) and "time" in result_item:
                        result_item["time"] = format_datetime(result_item["time"])

        return jsonify({"status": "success", "data": job_data})
