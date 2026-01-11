"""
TD3性能评估核心模块
"""
import numpy as np
import os
import sys
import traceback
from typing import Optional

# 添加lib目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'lib'))

from td3_inference_service_numpy import optimize_reactive_power
from pso_op_v2 import pso_op_v2
from .utils import sanitize_for_json, build_tunable_q_nodes
from .performance_config import translate_rating


def run_performance_evaluation(
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
    custom_thresholds: Optional[dict] = None,
    log_callback=None,
):
    """
    执行性能评估

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

    # 使用自定义阈值或全局阈值
    if custom_thresholds:
        # 将前端传入的 null 转换为 float('inf')
        thresholds = {}
        for label, threshold in custom_thresholds.items():
            if threshold is None or not isinstance(threshold, dict):
                thresholds[label] = {"voltage_error": float('inf'), "loss_error": float('inf')}
            else:
                thresholds[label] = {
                    "voltage_error": float('inf') if threshold.get("voltage_error") is None else threshold.get("voltage_error"),
                    "loss_error": float('inf') if threshold.get("loss_error") is None else threshold.get("loss_error"),
                }
    else:
        from .performance_config import get_thresholds_copy
        thresholds = get_thresholds_copy()

    def compute_rating(voltage_error: float, loss_error: float) -> str:
        # 处理 None 值或无效值
        if voltage_error is None or loss_error is None:
            return "unqualified"
        if not isinstance(voltage_error, (int, float)) or not isinstance(loss_error, (int, float)):
            return "unqualified"

        for label, threshold in thresholds.items():
            # 防御性检查：确保阈值本身不是 None
            if threshold is None or not isinstance(threshold, dict):
                continue
            v_threshold = threshold.get("voltage_error", float('inf'))
            l_threshold = threshold.get("loss_error", float('inf'))

            if voltage_error < v_threshold and loss_error < l_threshold:
                return label

        return "unqualified"

    # PSO参数
    num_particles = int(pso_params.get("num_particles", 30))
    max_iter = int(pso_params.get("max_iter", 50))
    w = float(pso_params.get("w", 0.8))
    c1 = float(pso_params.get("c1", 1.5))
    c2 = float(pso_params.get("c2", 1.5))
    pr = float(pso_params.get("pr", 1e-6))

    results = []
    successful_count = 0
    failed_count = 0

    # 节点统计（流式更新）
    node_names = None
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
                "avg_voltage_average_error": sanitize_for_json(float('inf')),
                "avg_loss_error": sanitize_for_json(float('inf')),
                "overall_rating": "unqualified",
                "node_summary": build_node_summary(),
            }

        avg_pre = sum_pre / ok_count
        avg_rl = sum_rl / ok_count
        avg_rl_red2 = sum_rl_red / ok_count
        avg_v_err2 = sum_v_err / ok_count
        avg_l_err2 = sum_l_err / ok_count

        avg_pso_2 = (sum_pso / pso_ok_count) if pso_ok_count else 0.0
        avg_pso_red2 = (sum_pso_red / ok_count) if ok_count else 0.0
        overall_rating = compute_rating(avg_v_err2, avg_l_err2)

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

    # 处理每个测试样本
    for i, sample in enumerate(test_samples):
        sample_time = sample["time"]
        ub = float(sample["ub"])
        bus_data = np.array(sample["bus"], dtype=float).reshape(-1, 3)

        # 添加日志：开始处理样本
        if log_callback:
            log_callback(f"----- 处理样本 {i+1}/{len(test_samples)} -----")
            log_callback(f"样本时间：{sample_time}")

        try:
            tunable_q_nodes = build_tunable_q_nodes(tunable_nodes, bus_data)
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

            # 执行TD3优化
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

            # 执行PSO优化
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

            # 检查 PSO 是否收敛
            pso_convergence = bool(pso_result.get("convergence", False))
            if not pso_convergence:
                # PSO 未收敛，跳过该样本
                if log_callback:
                    log_callback(f"PSO优化未收敛，跳过该样本")
                failed_count += 1
                continue

            # 提取优化结果
            rl_loss = float(td3_result["optimized_loss_rate"])
            rl_initial_loss = float(td3_result["initial_loss_rate"])
            pso_loss = float(pso_result["optimal_loss_rate"])
            pso_initial_loss = float(pso_result.get("initial_loss_rate", rl_initial_loss))
            pso_voltages = np.array(pso_result["final_voltages"], dtype=float)

            # 验证数据有效性
            if np.any(np.isnan(pso_voltages)) or np.any(np.isnan(rl_voltages)):
                if log_callback:
                    log_callback(f"电压数据包含无效值（NaN），跳过该样本")
                failed_count += 1
                continue

            # 计算降幅
            pre_loss = pso_initial_loss
            rl_reduction_pct = (pre_loss - rl_loss) / pre_loss * 100.0 if pre_loss else 0.0
            pso_reduction_pct = (pre_loss - pso_loss) / pre_loss * 100.0 if pre_loss else 0.0

            # 计算误差
            voltage_errors = np.abs(rl_voltages - pso_voltages) / pso_voltages * 100
            voltage_error = float(np.mean(voltage_errors))
            loss_error = float(np.abs(rl_loss - pso_loss) / pso_loss * 100) if pso_loss != 0 else float('inf')

            # 性能评级
            rating = compute_rating(voltage_error, loss_error)

            # 添加日志
            if log_callback:
                log_callback(f"优化前网损率：{pre_loss:.4f}%")
                log_callback(f"TD3优化网损率：{rl_loss:.4f}%（相对优化前降低：{rl_reduction_pct:.2f}%）")
                log_callback(f"PSO优化网损率：{pso_loss:.4f}%（相对优化前降低：{pso_reduction_pct:.2f}%）")
                log_callback(f"电压平均误差：{voltage_error:.4f}%，网损误差：{loss_error:.4f}%")
                log_callback(f"性能评估：{translate_rating(rating)}")

                # 输出节点无功值详情
                log_callback("无功调节策略对比：")
                log_callback(f"{'节点名称':<10} {'RL无功值(MVar)':<14} {'PSO无功值(MVar)':<14} {'无功上下限(MVar)':<20}")
                log_callback("-" * 70)
                td3_q_values = td3_result.get("optimized_q_values", [])
                pso_q_values = pso_result.get("optimal_params", [])
                for idx, node_info in enumerate(tunable_q_nodes):
                    node_name = node_info[3]
                    td3_q = td3_q_values[idx] if idx < len(td3_q_values) else 0.0
                    pso_q = pso_q_values[idx] if idx < len(pso_q_values) else 0.0
                    q_min = node_info[1]
                    q_max = node_info[2]
                    log_callback(f"{node_name:<12} {td3_q:<14.4f} {pso_q:<14.4f} {q_min:.4f} ~ {q_max:.4f}")

            # 记录结果
            result_item = {
                "time": sample_time,
                "success": True,
                "initial_loss_rate": round(rl_initial_loss, 4),
                "optimized_loss_rate": round(rl_loss, 4),
                "loss_reduction": round(rl_reduction_pct, 2),
                "voltage_violations": int(td3_result.get("voltage_violations", 0)),
                "optimized_voltages": rl_voltages.tolist(),
                "optimized_q_values": td3_result.get("optimized_q_values", []),
                "pso_convergence": True,
                "pso_initial_loss_rate": round(pso_initial_loss, 4),
                "pso_optimal_loss_rate": round(pso_loss, 4),
                "pso_optimal_params": pso_result.get("optimal_params", []),
                "pso_final_voltages": pso_result.get("final_voltages", []),
                "voltage_average_error": sanitize_for_json(round(voltage_error, 4) if not np.isinf(voltage_error) else voltage_error),
                "loss_error": sanitize_for_json(round(loss_error, 4) if not np.isinf(loss_error) else loss_error),
                "performance_rating": rating,
                "pre_optimization_loss": round(pre_loss, 4),
                "rl_relative_reduction": round(rl_reduction_pct, 2),
                "pso_relative_reduction": round(pso_reduction_pct, 2),
            }

            results.append(result_item)
            successful_count += 1

            # 更新节点无功值汇总
            for idx, q in enumerate(td3_result.get("optimized_q_values", [])):
                rl_q_sum_by_node[idx] += float(q)
                rl_q_count_by_node[idx] += 1
            for idx, q in enumerate(pso_result.get("optimal_params", [])):
                pso_q_sum_by_node[idx] += float(q)
                pso_q_count_by_node[idx] += 1

            ok_count += 1
            sum_pre += float(result_item["pre_optimization_loss"])
            sum_rl += float(result_item["optimized_loss_rate"])
            sum_rl_red += float(result_item["rl_relative_reduction"])
            sum_v_err += float(result_item["voltage_average_error"] or 0)
            sum_l_err += float(result_item["loss_error"] or 0)
            pso_ok_count += 1
            sum_pso += float(result_item["pso_optimal_loss_rate"])
            sum_pso_red += float(result_item["pso_relative_reduction"])

        except Exception as e:
            failed_count += 1
            # 添加日志：处理失败
            if log_callback:
                log_callback(f"❌ 处理失败：{str(e)}")
                stack_trace = traceback.format_exc()
                for line in stack_trace.split('\n'):
                    if line.strip():
                        log_callback(f"  {line}")
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
