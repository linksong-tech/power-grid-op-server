"""
路由模块初始化
"""
from flask import Blueprint

def register_routes(app):
    """注册所有路由到Flask应用"""
    from .health import health_bp
    from .flow_compute import flow_compute_bp
    from .pso_optimize import pso_optimize_bp
    
    app.register_blueprint(health_bp)
    app.register_blueprint(flow_compute_bp)
    app.register_blueprint(pso_optimize_bp)
