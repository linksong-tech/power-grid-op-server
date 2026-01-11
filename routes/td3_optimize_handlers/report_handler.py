"""
TD3评估报告路由处理器
"""
from flask import jsonify
import numpy as np
import traceback

from routes.td3_config import batch_jobs, batch_jobs_lock


def get_evaluation_report(job_id: str):
    """
    获取性能评估汇总报告

    返回格式:
    {
        "status": "success",
        "data": {
            "summary": {...},
            "chart_data": {...},
            "details": [...],
            "node_summary": [...]
        }
    }
    """
    try:
        with batch_jobs_lock:
            job = batch_jobs.get(job_id)
            if not job:
                return jsonify({"status": "error", "message": "任务不存在"}), 404

            if job["status"] != "completed":
                return jsonify({"status": "error", "message": "任务尚未完成"}), 400

            result = job.get("result")
            if not result:
                return jsonify({"status": "error", "message": "无评估结果"}), 404

        evaluation = result.get("evaluation", {})
        results = result.get("results", [])

        # 计算有效样本数
        valid_results = [r for r in results if r.get("success")]
        valid_samples = len(valid_results)

        if valid_samples == 0:
            return jsonify({"status": "error", "message": "无有效评估结果"}), 404

        # 汇总统计
        summary = {
            "valid_samples": valid_samples,
            "total_samples": len(results),
            "failed_samples": len(results) - valid_samples,
            "avg_pre_optimization_loss": evaluation.get("avg_pre_optimization_loss", 0),
            "avg_rl_loss": evaluation.get("avg_rl_loss", 0),
            "avg_pso_loss": evaluation.get("avg_pso_loss", 0),
            "avg_rl_reduction": evaluation.get("avg_rl_relative_reduction", 0),
            "avg_pso_reduction": evaluation.get("avg_pso_relative_reduction", 0),
            "avg_voltage_error": evaluation.get("avg_voltage_average_error", 0),
            "avg_loss_error": evaluation.get("avg_loss_error", 0),
            "overall_rating": evaluation.get("overall_rating", "unqualified"),
        }

        # 性能评级分布统计
        rating_counts = {"excellent": 0, "good": 0, "qualified": 0, "unqualified": 0}
        for r in valid_results:
            rating = r.get("performance_rating", "unqualified")
            if rating in rating_counts:
                rating_counts[rating] += 1

        summary["performance_distribution"] = rating_counts

        # 准备图表数据
        voltage_errors = [r.get("voltage_average_error", 0) for r in valid_results]
        loss_errors = [r.get("loss_error", 0) for r in valid_results]
        rl_losses = [r.get("optimized_loss_rate", 0) for r in valid_results]
        pso_losses = [r.get("pso_optimal_loss_rate", 0) for r in valid_results if r.get("pso_convergence")]

        chart_data = {
            "error_distribution": {
                "voltage_errors": voltage_errors,
                "loss_errors": loss_errors,
                "voltage_stats": {
                    "mean": float(np.mean(voltage_errors)) if voltage_errors else 0,
                    "median": float(np.median(voltage_errors)) if voltage_errors else 0,
                    "max": float(np.max(voltage_errors)) if voltage_errors else 0,
                    "min": float(np.min(voltage_errors)) if voltage_errors else 0,
                },
                "loss_stats": {
                    "mean": float(np.mean(loss_errors)) if loss_errors else 0,
                    "median": float(np.median(loss_errors)) if loss_errors else 0,
                    "max": float(np.max(loss_errors)) if loss_errors else 0,
                    "min": float(np.min(loss_errors)) if loss_errors else 0,
                }
            },
            "loss_scatter": {
                "rl_losses": rl_losses,
                "pso_losses": pso_losses,
                "correlation": float(np.corrcoef(rl_losses, pso_losses)[0, 1]) if len(rl_losses) > 1 and len(pso_losses) > 1 else 0,
            }
        }

        return jsonify({
            "status": "success",
            "data": {
                "summary": summary,
                "chart_data": chart_data,
                "details": valid_results[:100],  # 限制返回前100条详细结果
                "node_summary": evaluation.get("node_summary", [])
            }
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"获取报告失败: {str(e)}"
        }), 500
