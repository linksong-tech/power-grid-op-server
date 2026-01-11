"""
TD3批量任务管理模块
"""
from datetime import datetime
from routes.td3_config import batch_jobs, batch_jobs_lock
from .utils import get_timezone


def append_job_log(job_id: str, message: str):
    """向任务添加日志"""
    tz = get_timezone()
    ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {message}"
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            print(f"[WARN] Job {job_id} not found in append_job_log", flush=True)
            return
        job["logs"].append(entry)
        # 限制日志长度
        if len(job["logs"]) > 2000:
            job["logs"] = job["logs"][-2000:]
        job["updated_at"] = datetime.now(tz).isoformat()
    # 调试输出（Docker环境）
    print(f"[DEBUG] Job {job_id}: {message}", flush=True)


def update_job(job_id: str, patch: dict):
    """更新任务状态"""
    tz = get_timezone()
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            print(f"[WARN] Job {job_id} not found in update_job", flush=True)
            return
        job.update(patch)
        job["updated_at"] = datetime.now(tz).isoformat()
    # 调试输出（Docker环境）- 只输出关键字段
    debug_info = {k: v for k, v in patch.items() if k in ['status', 'progress', 'processed', 'total', 'message']}
    if debug_info:
        print(f"[DEBUG] Job {job_id} updated: {debug_info}", flush=True)


def get_job(job_id: str):
    """获取任务信息"""
    with batch_jobs_lock:
        return batch_jobs.get(job_id)
