"""
TD3性能评估配置模块
"""
import threading

# 全局性能阈值配置（可通过API动态修改）
# 注意：unqualified 使用 float('inf') 表示无限制，JSON 序列化时会转为 null
PERFORMANCE_THRESHOLDS = {
    "excellent": {"voltage_error": 0.5, "loss_error": 3.0},
    "good": {"voltage_error": 1.0, "loss_error": 4.0},
    "qualified": {"voltage_error": 2.0, "loss_error": 5.0},
    "unqualified": {"voltage_error": float('inf'), "loss_error": float('inf')},
}

# 用于线程安全的阈值访问
thresholds_lock = threading.Lock()

# 评级英文到中文的映射
RATING_TRANSLATION = {
    "excellent": "优秀",
    "good": "良好",
    "qualified": "合格",
    "unqualified": "不合格"
}


def compute_rating(voltage_error: float, loss_error: float) -> str:
    """计算性能评级"""
    with thresholds_lock:
        for label, threshold in PERFORMANCE_THRESHOLDS.items():
            if voltage_error <= threshold["voltage_error"] and loss_error <= threshold["loss_error"]:
                return label
    return "unqualified"


def translate_rating(rating: str) -> str:
    """将英文评级转换为中文"""
    return RATING_TRANSLATION.get(rating, rating)


def get_thresholds_copy():
    """获取阈值的线程安全副本"""
    with thresholds_lock:
        return PERFORMANCE_THRESHOLDS.copy()


def update_thresholds(new_thresholds: dict):
    """更新全局阈值配置"""
    with thresholds_lock:
        PERFORMANCE_THRESHOLDS.update(new_thresholds)
