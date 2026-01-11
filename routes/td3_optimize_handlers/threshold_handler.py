"""
TD3性能阈值配置路由处理器
"""
from flask import jsonify, request
import traceback

from .performance_config import get_thresholds_copy, update_thresholds, PERFORMANCE_THRESHOLDS
from .utils import sanitize_for_json


def get_performance_thresholds():
    """获取当前性能阈值配置（将 float('inf') 转换为 null）"""
    thresholds = get_thresholds_copy()
    # 深拷贝并转换 inf 为 null
    thresholds_for_json = {}
    for label, threshold in thresholds.items():
        thresholds_for_json[label] = {
            "voltage_error": sanitize_for_json(threshold["voltage_error"]),
            "loss_error": sanitize_for_json(threshold["loss_error"]),
        }
    return jsonify({"status": "success", "data": thresholds_for_json})


def update_performance_thresholds():
    """更新性能阈值配置"""
    try:
        data = request.get_json()

        # 验证数据格式
        required_levels = ["excellent", "good", "qualified", "unqualified"]
        for level in required_levels:
            if level not in data:
                return jsonify({
                    "status": "error",
                    "message": f"缺少等级: {level}"
                }), 400

            if "voltage_error" not in data[level] or "loss_error" not in data[level]:
                return jsonify({
                    "status": "error",
                    "message": f"等级 {level} 缺少必需字段"
                }), 400

        # 更新全局阈值
        update_thresholds(data)

        return jsonify({"status": "success", "success": True})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"更新阈值失败: {str(e)}"
        }), 500
