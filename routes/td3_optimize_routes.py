"""
TD3优化执行相关路由
"""
from flask import jsonify, request
import json
import numpy as np
import os
import sys
import threading
import uuid
from datetime import datetime
import traceback
from typing import Optional

# 添加lib目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from td3_inference_service import optimize_reactive_power, batch_optimize
from pso_op_v2 import pso_op_v2
from routes.td3_config import RESULTS_DIR, TRAINING_DATA_DIR, batch_jobs, batch_jobs_lock
from routes.td3_data_reader import load_test_samples


def find_model_path(model_path, line_name):
    """
    查找模型文件路径

    Args:
        model_path: 模型文件名
        line_name: 线路名称（必填，用于在指定线路的agent目录中查找）

    Returns:
        模型完整路径，如果未找到则返回None
    """
    # 只在指定线路的 TRAINING_DATA_DIR/{line_name}/agent/ 中查找
    if not line_name:
        return None

    model_full_path = os.path.join(TRAINING_DATA_DIR, line_name, 'agent', model_path)
    if os.path.exists(model_full_path):
        return model_full_path

    return None


PERFORMANCE_THRESHOLDS = {
    "优秀": {"voltage_error": 0.5, "loss_error": 3.0},
    "良好": {"voltage_error": 1.0, "loss_error": 4.0},
    "合格": {"voltage_error": 2.0, "loss_error": 5.0},
    "不合格": {"voltage_error": float("inf"), "loss_error": float("inf")},
}


def _compute_rating(voltage_error: float, loss_error: float) -> str:
    for label, threshold in PERFORMANCE_THRESHOLDS.items():
        if voltage_error < threshold["voltage_error"] and loss_error < threshold["loss_error"]:
            return label
    return "不合格"


def _append_job_log(job_id: str, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {message}"
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            return
        job["logs"].append(entry)
        # 限制日志长度
        if len(job["logs"]) > 2000:
            job["logs"] = job["logs"][-2000:]
        job["updated_at"] = datetime.now().isoformat()


def _update_job(job_id: str, patch: dict):
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            return
        job.update(patch)
        job["updated_at"] = datetime.now().isoformat()


def _build_tunable_q_nodes(tunable_nodes, bus_data: np.ndarray):
    """
    支持两种格式：
    - [node_index, q_min, q_max, name]：直接使用（node_index 为 0-based）
    - [node_no, capacity, name]：根据 bus_data 计算 q_min/q_max（node_no 为 1-based）
    """
    built = []
    for node in tunable_nodes:
        if not isinstance(node, (list, tuple)):
            raise ValueError("tunableNodes 格式错误：元素必须为数组")
        if len(node) == 4:
            node_idx = int(node[0])
            q_min = float(node[1])
            q_max = float(node[2])
            name = str(node[3])
            built.append((node_idx, q_min, q_max, name))
            continue
        if len(node) == 3:
            node_no = int(node[0])
            capacity = float(node[1])
            name = str(node[2])
            node_idx = node_no - 1
            if node_idx < 0 or node_idx >= bus_data.shape[0]:
                raise ValueError(f"tunableNodes 节点号超出范围: {node_no}")
            p_current = float(abs(bus_data[node_idx, 1]))
            q_max = float(np.sqrt(max(0.0, capacity**2 - p_current**2)))
            q_min = -q_max
            built.append((node_idx, q_min, q_max, name))
            continue
        raise ValueError("tunableNodes 格式错误：仅支持长度为3或4的数组")
    return built


def _run_perf_evaluation(
    *,
    test_samples,
    branch_data: np.ndarray,
    voltage_limits,
    key_nodes,
    tunable_nodes,
    model_full_path: str,
    sb: float,
    pso_params: Optional[dict] = None,
    progress_callback=None,
    update_callback=None,
):
    """
    返回：
    {
      'success': bool,
      'total_samples': int,
      'successful_optimizations': int,
      'failed_optimizations': int,
      'results': [...],            # 每个断面明细（含TD3+PSO+误差+评级）
      'evaluation': { ... },       # 汇总&节点平均
    }
    """
    pso_params = pso_params or {}

    num_particles = int(pso_params.get("num_particles", 30))
    max_iter = int(pso_params.get("max_iter", 50))
    w = float(pso_params.get("w", 0.8))
    c1 = float(pso_params.get("c1", 1.5))
    c2 = float(pso_params.get("c2", 1.5))
    pr = float(pso_params.get("pr", 1e-6))

    results = []
    successful_count = 0
    failed_count = 0

    node_names = None
    # 节点统计（流式更新）
    rl_q_sum_by_node = None
    rl_q_count_by_node = None
    pso_q_sum_by_node = None
    pso_q_count_by_node = None
    q_min_by_node = None
    q_max_by_node = None

    # 汇总统计（流式更新）
    ok_count = 0
    pso_ok_count = 0
    sum_pre = 0.0
    sum_rl = 0.0
    sum_pso = 0.0
    sum_rl_red = 0.0
    sum_pso_red = 0.0
    sum_v_err = 0.0
    sum_l_err = 0.0

    def build_node_summary():
        if not node_names:
            return []
        summary = []
        for idx, name in enumerate(node_names):
            rl_avg = (rl_q_sum_by_node[idx] / rl_q_count_by_node[idx]) if rl_q_count_by_node[idx] else 0.0
            pso_avg = (pso_q_sum_by_node[idx] / pso_q_count_by_node[idx]) if pso_q_count_by_node[idx] else 0.0
            summary.append({
                "node_index": idx,
                "node_name": name,
                "rl_avg_q": round(float(rl_avg), 4),
                "pso_avg_q": round(float(pso_avg), 4),
                "q_min": round(float(q_min_by_node[idx]), 4) if q_min_by_node[idx] is not None else 0.0,
                "q_max": round(float(q_max_by_node[idx]), 4) if q_max_by_node[idx] is not None else 0.0,
            })
        return summary

    def build_evaluation_snapshot():
        if ok_count <= 0:
            return {
                "avg_pre_optimization_loss": 0.0,
                "avg_rl_loss": 0.0,
                "avg_pso_loss": 0.0,
                "avg_rl_relative_reduction": 0.0,
                "avg_pso_relative_reduction": 0.0,
                "avg_voltage_average_error": float("inf"),
                "avg_loss_error": float("inf"),
                "overall_rating": "不合格",
                "node_summary": build_node_summary(),
            }

        avg_pre = sum_pre / ok_count
        avg_rl = sum_rl / ok_count
        avg_rl_red2 = sum_rl_red / ok_count
        avg_v_err2 = sum_v_err / ok_count
        avg_l_err2 = sum_l_err / ok_count

        avg_pso_2 = (sum_pso / pso_ok_count) if pso_ok_count else 0.0
        avg_pso_red2 = (sum_pso_red / ok_count) if ok_count else 0.0
        overall_rating = _compute_rating(avg_v_err2, avg_l_err2)

        return {
            "avg_pre_optimization_loss": round(float(avg_pre), 4),
            "avg_rl_loss": round(float(avg_rl), 4),
            "avg_pso_loss": round(float(avg_pso_2), 4),
            "avg_rl_relative_reduction": round(float(avg_rl_red2), 2),
            "avg_pso_relative_reduction": round(float(avg_pso_red2), 2),
            "avg_voltage_average_error": round(float(avg_v_err2), 4),
            "avg_loss_error": round(float(avg_l_err2), 4),
            "overall_rating": overall_rating,
            "node_summary": build_node_summary(),
        }

    for i, sample in enumerate(test_samples):
        sample_time = sample["time"]
        ub = float(sample["ub"])
        bus_data = np.array(sample["bus"], dtype=float).reshape(-1, 3)

        try:
            tunable_q_nodes = _build_tunable_q_nodes(tunable_nodes, bus_data)
            if node_names is None:
                node_names = [n[3] for n in tunable_q_nodes]
                rl_q_sum_by_node = [0.0 for _ in tunable_q_nodes]
                rl_q_count_by_node = [0 for _ in tunable_q_nodes]
                pso_q_sum_by_node = [0.0 for _ in tunable_q_nodes]
                pso_q_count_by_node = [0 for _ in tunable_q_nodes]
                q_min_by_node = [float(n[1]) for n in tunable_q_nodes]
                q_max_by_node = [float(n[2]) for n in tunable_q_nodes]
            else:
                # 更新节点上下限（不同断面可能不同，取全局覆盖范围）
                for idx, n in enumerate(tunable_q_nodes):
                    q_min_by_node[idx] = min(q_min_by_node[idx], float(n[1]))
                    q_max_by_node[idx] = max(q_max_by_node[idx], float(n[2]))

            td3_result = optimize_reactive_power(
                bus_data=bus_data,
                branch_data=branch_data,
                voltage_limits=tuple(voltage_limits),
                key_nodes=key_nodes,
                tunable_q_nodes=tunable_q_nodes,
                model_path=model_full_path,
                ub=ub,
                sb=sb,
            )

            if not td3_result.get("success"):
                raise RuntimeError(td3_result.get("error", "TD3优化失败"))

            rl_voltages = np.array(td3_result["optimized_voltages"], dtype=float)

            pso_result = pso_op_v2(
                bus_data,
                branch_data,
                tunable_q_nodes,
                num_particles=num_particles,
                max_iter=max_iter,
                w=w,
                c1=c1,
                c2=c2,
                v_min=float(voltage_limits[0]),
                v_max=float(voltage_limits[1]),
                SB=sb,
                UB=ub,
                pr=pr,
            )

            pso_convergence = bool(pso_result.get("convergence", False))
            rl_loss = float(td3_result["optimized_loss_rate"])
            rl_initial_loss = float(td3_result["initial_loss_rate"])
            pso_loss = float(pso_result["optimal_loss_rate"]) if pso_convergence else None
            pso_initial_loss = float(pso_result.get("initial_loss_rate", rl_initial_loss))

            pre_loss = pso_initial_loss
            rl_reduction_pct = (pre_loss - rl_loss) / pre_loss * 100.0 if pre_loss else 0.0
            pso_reduction_pct = (pre_loss - pso_loss) / pre_loss * 100.0 if (pre_loss and pso_loss is not None) else 0.0

            voltage_error = None
            loss_error = None
            rating = None

            if pso_convergence and pso_loss is not None:
                pso_voltages = np.array(pso_result["final_voltages"], dtype=float)
                voltage_error = float(np.mean(np.abs(rl_voltages - pso_voltages)) / ub * 100.0) if ub else 0.0
                loss_error = float(abs(rl_loss - pso_loss))
                rating = _compute_rating(voltage_error, loss_error)
            else:
                voltage_error = float("inf")
                loss_error = float("inf")
                rating = "不合格"

            # 节点无功值汇总（平均）
            for idx, q in enumerate(td3_result.get("optimized_q_values", [])):
                rl_q_sum_by_node[idx] += float(q)
                rl_q_count_by_node[idx] += 1
            if pso_convergence:
                for idx, q in enumerate(pso_result.get("optimal_params", [])):
                    pso_q_sum_by_node[idx] += float(q)
                    pso_q_count_by_node[idx] += 1

            results.append({
                "time": sample_time,
                "success": True,
                # TD3
                "initial_loss_rate": round(rl_initial_loss, 4),
                "optimized_loss_rate": round(rl_loss, 4),
                "loss_reduction": round(rl_reduction_pct, 2),
                "voltage_violations": int(td3_result.get("voltage_violations", 0)),
                "optimized_voltages": rl_voltages.tolist(),
                "optimized_q_values": td3_result.get("optimized_q_values", []),
                # PSO
                "pso_convergence": pso_convergence,
                "pso_initial_loss_rate": round(pso_initial_loss, 4),
                "pso_optimal_loss_rate": round(pso_loss, 4) if pso_loss is not None else None,
                "pso_optimal_params": pso_result.get("optimal_params", []),
                "pso_final_voltages": pso_result.get("final_voltages", []) if pso_convergence else None,
                # 误差&评级（用于性能评估UI）
                "voltage_average_error": round(voltage_error, 4),
                "loss_error": round(loss_error, 4),
                "performance_rating": rating,
                # 额外（便于前端展示降幅）
                "pre_optimization_loss": round(pre_loss, 4),
                "rl_relative_reduction": round(rl_reduction_pct, 2),
                "pso_relative_reduction": round(pso_reduction_pct, 2),
            })

            successful_count += 1

            ok_count += 1
            sum_pre += float(pre_loss)
            sum_rl += float(rl_loss)
            sum_rl_red += float(rl_reduction_pct)
            sum_v_err += float(voltage_error)
            sum_l_err += float(loss_error)
            if pso_convergence and pso_loss is not None:
                pso_ok_count += 1
                sum_pso += float(pso_loss)
            sum_pso_red += float(pso_reduction_pct)
        except Exception as e:
            failed_count += 1
            results.append({
                "time": sample_time,
                "success": False,
                "error": str(e),
            })

        if update_callback:
            update_callback({
                "success": True,
                "total_samples": len(test_samples),
                "successful_optimizations": successful_count,
                "failed_optimizations": failed_count,
                "results": results,
                "evaluation": build_evaluation_snapshot(),
            })

        if progress_callback:
            progress_callback(i + 1, len(test_samples), sample_time)

    return {
        "success": True,
        "total_samples": len(test_samples),
        "successful_optimizations": successful_count,
        "failed_optimizations": failed_count,
        "results": results,
        "evaluation": build_evaluation_snapshot(),
    }


def td3_optimize():
    """
    执行TD3强化学习无功优化

    请求体格式:
    {
        "busData": [[节点号, 有功P, 无功Q], ...],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "modelPath": "模型文件名",
        "lineName": "线路名称（必填，用于在指定线路的agent目录中查找模型）",
        "parameters": {
            "UB": 10.38,
            "SB": 10
        }
    }
    """
    try:
        data = request.get_json()

        # 验证必需参数
        required_fields = ['busData', 'branchData', 'voltageLimits', 'keyNodes', 'tunableNodes', 'modelPath', 'lineName']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必需参数: {field}'
                }), 400

        # 获取参数
        bus_data = np.array(data['busData'])
        branch_data = np.array(data['branchData'])
        voltage_limits = tuple(data['voltageLimits'])
        key_nodes = data['keyNodes']
        tunable_nodes = [tuple(node) for node in data['tunableNodes']]
        model_path = data['modelPath']
        line_name = data['lineName']  # 必填参数

        # 获取物理参数
        params = data.get('parameters', {})
        ub = params.get('UB', 10.38)
        sb = params.get('SB', 10)

        # 检查模型文件是否存在
        model_full_path = find_model_path(model_path, line_name)
        if not model_full_path:
            return jsonify({
                'status': 'error',
                'message': f'模型文件不存在: {model_path}'
            }), 404
        
        # 执行TD3优化
        result = optimize_reactive_power(
            bus_data=bus_data,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_nodes,
            model_path=model_full_path,
            ub=ub,
            sb=sb
        )
        
        if not result['success']:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'TD3优化失败')
            }), 500
        
        # 保存优化结果
        result_file = os.path.join(RESULTS_DIR, f'td3_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            save_result = result.copy()
            if 'optimized_voltages' in save_result:
                save_result['optimized_voltages'] = save_result['optimized_voltages'].tolist()
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        # 返回结果（转换numpy数组）
        response_result = result.copy()
        if 'optimized_voltages' in response_result:
            response_result['optimized_voltages'] = response_result['optimized_voltages'].tolist()
        
        return jsonify({
            'status': 'success',
            'data': response_result
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'TD3优化失败: {str(e)}'
        }), 500


def td3_batch_optimize():
    """
    批量执行TD3优化

    请求体格式:
    {
        "testSamples": [
            {
                "time": "2024-12-29 08:00",
                "ub": 10.0,
                "bus": [节点号1, P1, Q1, 节点号2, P2, Q2, ...]
            },
            ...
        ],
        "branchData": [[线路号, 首节点, 末节点, 电阻R, 电抗X], ...],
        "voltageLimits": [v_min, v_max],
        "keyNodes": [节点索引1, 节点索引2, ...],
        "tunableNodes": [[节点索引, Q_min, Q_max, "节点名"], ...],
        "modelPath": "模型文件名",
        "lineName": "线路名称（必填，用于在指定线路的agent目录中查找模型）",
        "parameters": {
            "SB": 10
        }
    }
    """
    try:
        data = request.get_json()

        is_async = bool(data.get("async", False))

        # 验证必需参数（testSamples 可由后端从已上传数据中读取）
        required_fields = ['branchData', 'voltageLimits', 'keyNodes', 'tunableNodes', 'modelPath', 'lineName']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'缺少必需参数: {field}'}), 400

        # 获取参数
        line_name = data['lineName']  # 必填参数
        test_samples = data.get('testSamples')
        if not test_samples:
            # 从 TRAINING_DATA_DIR/{line_name}/test 读取（由 upload-powerdata 上传）
            test_samples = load_test_samples(line_name, TRAINING_DATA_DIR)
        branch_data = np.array(data['branchData'])
        voltage_limits = tuple(data['voltageLimits'])
        key_nodes = data['keyNodes']
        tunable_nodes = data['tunableNodes']
        model_path = data['modelPath']

        # 获取物理参数
        params = data.get('parameters', {}) or {}
        sb = float(params.get('SB', 10))
        pso_params = params.get('psoParameters', {}) or {}

        # 检查模型文件是否存在
        model_full_path = find_model_path(model_path, line_name)
        if not model_full_path:
            return jsonify({
                'status': 'error',
                'message': f'模型文件不存在: {model_path}'
            }), 404
        
        if is_async:
            job_id = uuid.uuid4().hex
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
                            "avg_voltage_average_error": float("inf"),
                            "avg_loss_error": float("inf"),
                            "overall_rating": "不合格",
                            "node_summary": [],
                        },
                    },
                    "error": None,
                    "started_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

            _append_job_log(job_id, f"开始批量性能评估：样本数={len(test_samples)}")

            def progress_callback(processed, total, sample_time):
                progress = int(processed / total * 100) if total else 0
                _update_job(job_id, {
                    "processed": processed,
                    "total": total,
                    "progress": progress,
                    "current_sample_time": sample_time,
                    "message": f"处理中 {processed}/{total}",
                })
                _append_job_log(job_id, f"完成断面 {processed}/{total}: {sample_time}")

            def run_in_background():
                try:
                    result = _run_perf_evaluation(
                        test_samples=test_samples,
                        branch_data=branch_data,
                        voltage_limits=voltage_limits,
                        key_nodes=key_nodes,
                        tunable_nodes=tunable_nodes,
                        model_full_path=model_full_path,
                        sb=sb,
                        pso_params=pso_params,
                        progress_callback=progress_callback,
                        update_callback=lambda partial: _update_job(job_id, {"result": partial}),
                    )
                    _update_job(job_id, {
                        "status": "completed",
                        "progress": 100,
                        "message": "评估完成",
                        "result": result,
                    })
                    _append_job_log(job_id, "评估完成")

                    # 保存结果文件
                    result_file = os.path.join(RESULTS_DIR, f'td3_perf_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{job_id}.json')
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    _update_job(job_id, {
                        "status": "failed",
                        "message": f"评估失败: {str(e)}",
                        "error": str(e),
                    })
                    _append_job_log(job_id, f"评估失败: {str(e)}")
                    print(traceback.format_exc())

            t = threading.Thread(target=run_in_background)
            t.daemon = True
            t.start()

            return jsonify({
                "status": "success",
                "data": {"job_id": job_id},
            })

        # 同步执行（兼容旧调用）
        result = _run_perf_evaluation(
            test_samples=test_samples,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_nodes=tunable_nodes,
            model_full_path=model_full_path,
            sb=sb,
            pso_params=pso_params,
        )

        # 保存结果
        result_file = os.path.join(RESULTS_DIR, f'td3_perf_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'success', 'data': result})
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'批量优化失败: {str(e)}'
        }), 500


def get_td3_batch_status(job_id: str):
    """获取批量评估任务状态（供前端短轮询）"""
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "任务不存在"}), 404
        return jsonify({"status": "success", "data": job})
