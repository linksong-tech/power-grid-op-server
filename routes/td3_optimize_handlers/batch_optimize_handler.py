"""
TD3批量优化路由处理器
"""
from flask import jsonify, request
import json
import numpy as np
import os
import sys
import threading
import uuid
import traceback
from datetime import datetime

from lib.line_service import line_service
from routes.td3_config import batch_jobs, batch_jobs_lock
from .model_finder import find_model_path_by_line_id, load_test_samples_by_line_id
from .performance_evaluator import run_performance_evaluation
from .job_manager import append_job_log, update_job
from .utils import get_timezone, sanitize_for_json


def _handle_async_batch_optimize(
    line_id, test_samples, branch_data, voltage_limits,
    key_nodes, tunable_nodes, model_full_path, sb,
    pso_params, custom_thresholds
):
    """处理异步批量优化"""
    job_id = uuid.uuid4().hex
    tz = get_timezone()

    with batch_jobs_lock:
        batch_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "processed": 0,
            "total": len(test_samples),
            "current_sample_time": None,
            "message": "任务已启动",
            "logs": [],
            "result": {
                "success": True,
                "total_samples": len(test_samples),
                "successful_optimizations": 0,
                "failed_optimizations": 0,
                "results": [],
                "evaluation": {
                    "avg_pre_optimization_loss": 0.0,
                    "avg_rl_loss": 0.0,
                    "avg_pso_loss": 0.0,
                    "avg_rl_relative_reduction": 0.0,
                    "avg_pso_relative_reduction": 0.0,
                    "avg_voltage_average_error": sanitize_for_json(float('inf')),
                    "avg_loss_error": sanitize_for_json(float('inf')),
                    "overall_rating": "unqualified",
                    "node_summary": [],
                },
            },
            "error": None,
            "started_at": datetime.now(tz).isoformat(),
            "updated_at": datetime.now(tz).isoformat(),
        }

    append_job_log(job_id, f"开始批量性能评估：样本数={len(test_samples)}")

    def progress_callback(processed, total, sample_time):
        try:
            progress = int(processed / total * 100) if total else 0
            update_job(job_id, {
                "processed": processed,
                "total": total,
                "progress": progress,
                "current_sample_time": sample_time,
                "message": f"处理中 {processed}/{total}",
            })
            append_job_log(job_id, f"完成断面 {processed}/{total}: {sample_time}")
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            print(f"[ERROR] progress_callback failed: {e}", flush=True)
            traceback.print_exc()

    def update_callback_wrapper(partial):
        try:
            update_job(job_id, {"result": partial})
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            print(f"[ERROR] update_callback failed: {e}", flush=True)
            traceback.print_exc()

    def log_callback_wrapper(message):
        try:
            append_job_log(job_id, message)
        except Exception as e:
            print(f"[ERROR] log_callback failed: {e}", flush=True)
            traceback.print_exc()

    def run_in_background():
        try:
            print(f"[INFO] Starting performance evaluation for job {job_id}", flush=True)
            result = run_performance_evaluation(
                test_samples=test_samples,
                branch_data=branch_data,
                voltage_limits=voltage_limits,
                key_nodes=key_nodes,
                tunable_nodes=tunable_nodes,
                model_full_path=model_full_path,
                sb=sb,
                pso_params=pso_params,
                progress_callback=progress_callback,
                update_callback=update_callback_wrapper,
                custom_thresholds=custom_thresholds,
                log_callback=log_callback_wrapper,
            )
            print(f"[INFO] Performance evaluation completed for job {job_id}", flush=True)
            update_job(job_id, {
                "status": "completed",
                "progress": 100,
                "message": "评估完成",
                "result": result,
            })
            append_job_log(job_id, "评估完成")

            # 保存结果文件
            line_dir = line_service._get_line_dir(line_id)
            perf_eval_dir = os.path.join(line_dir, 'perf_eval')
            os.makedirs(perf_eval_dir, exist_ok=True)
            result_file = os.path.join(perf_eval_dir, f'perf_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{job_id}.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
            print(f"[INFO] Result saved to {result_file}", flush=True)
        except Exception as e:
            error_msg = f"评估失败: {str(e)}"
            error_trace = traceback.format_exc()
            print(f"[ERROR] {error_msg}", flush=True)
            print(error_trace, flush=True)
            update_job(job_id, {
                "status": "failed",
                "message": error_msg,
                "error": str(e),
            })
            append_job_log(job_id, error_msg)

    t = threading.Thread(target=run_in_background)
    t.daemon = True
    t.start()

    return jsonify({
        "status": "success",
        "data": {"job_id": job_id},
    })


def td3_batch_optimize():
    """
    批量执行TD3优化

    请求体格式:
    {
        "testSamples": [...],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "modelPath": "模型文件名",
        "lineId": "线路ID（必填）",
        "parameters": {
            "SB": 10
        }
    }
    """
    try:
        data = request.get_json()
        is_async = bool(data.get("async", False))

        # 验证必需参数
        required_fields = ['branchData', 'voltageLimits', 'keyNodes', 'tunableNodes', 'modelPath', 'lineId']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'缺少必需参数: {field}'}), 400

        # 获取参数
        line_id = data['lineId']
        test_samples = data.get('testSamples')
        if not test_samples:
            test_samples = load_test_samples_by_line_id(line_id)
        branch_data = np.array(data['branchData'])
        voltage_limits = tuple(data['voltageLimits'])
        key_nodes = data['keyNodes']
        tunable_nodes = data['tunableNodes']
        model_path = data['modelPath']

        # 获取物理参数
        params = data.get('parameters', {}) or {}
        sb = float(params.get('SB', 10))
        pso_params = params.get('psoParameters', {}) or {}

        # 获取性能阈值
        custom_thresholds = data.get('performanceThresholds')

        # 检查模型文件是否存在
        model_full_path = find_model_path_by_line_id(model_path, line_id)
        if not model_full_path:
            return jsonify({
                'status': 'error',
                'message': f'模型文件不存在: {model_path}'
            }), 404

        if is_async:
            return _handle_async_batch_optimize(
                line_id, test_samples, branch_data, voltage_limits,
                key_nodes, tunable_nodes, model_full_path, sb,
                pso_params, custom_thresholds
            )

        # 同步执行
        result = run_performance_evaluation(
            test_samples=test_samples,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_nodes=tunable_nodes,
            model_full_path=model_full_path,
            sb=sb,
            pso_params=pso_params,
            custom_thresholds=custom_thresholds,
        )

        # 保存结果
        line_dir = line_service._get_line_dir(line_id)
        perf_eval_dir = os.path.join(line_dir, 'perf_eval')
        os.makedirs(perf_eval_dir, exist_ok=True)
        result_file = os.path.join(perf_eval_dir, f'perf_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'批量优化失败: {str(e)}'
        }), 500
