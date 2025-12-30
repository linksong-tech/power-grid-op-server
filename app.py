from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)  # 启用CORS支持跨域请求

# 配置
app.config['JSON_AS_ASCII'] = False  # 支持中文字符
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 创建数据存储目录
DATA_DIR = 'data'
RESULTS_DIR = 'results'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 注册路由
from routes import register_routes
register_routes(app)

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': '接口不存在'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误'
    }), 500

if __name__ == '__main__':
    port = 5000
    if len(sys.argv) > 1 and '--port' in sys.argv:
        try:
            port_idx = sys.argv.index('--port')
            port = int(sys.argv[port_idx + 1])
        except (ValueError, IndexError):
            pass
    
    print("启动电力潮流计算服务...")
    print(f"服务地址: http://localhost:{port}")
    print("API文档:")
    print("  GET  /api/health - 健康检查")
    print("  POST /api/flow-compute/parameters - 保存参数")
    print("  GET  /api/flow-compute/parameters - 获取参数")
    print("  POST /api/flow-compute/calculate - 执行计算")
    print("  GET  /api/flow-compute/template - 获取模板")
    print("  GET  /api/flow-compute/results - 获取结果列表")
    print("  GET  /api/flow-compute/results/<filename> - 获取结果详情")
    print("  GET  /api/flow-compute/export/<filename> - 导出结果")
    print("  GET  /api/flow-compute/oc-source - 从宿主机接口获取运行工况数据")
    print("  GET  /api/flow-compute/oc-source-local - 从本地JSON文件获取运行工况数据")
    print("  POST /api/pso-optimize/parameters - 保存PSO参数")
    print("  GET  /api/pso-optimize/parameters - 获取PSO参数")
    print("  POST /api/pso-optimize/optimize - 执行PSO优化")
    print("  POST /api/pso-optimize/optimize/v2 - 执行PSO优化(v2)")
    print("  GET  /api/pso-optimize/template - 获取PSO模板")
    print("  GET  /api/pso-optimize/results - 获取PSO结果列表")
    print("\n注意: 如果5000端口被占用，请使用 --port 参数指定其他端口")
    print("例如: python app.py --port 5001")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ 端口 {port} 已被占用")
            print("解决方案:")
            print("1. 使用其他端口: python app.py --port 5001")
            print("2. 关闭占用端口的程序")
            print("3. 在macOS上，可在 系统偏好设置 -> 共享 中关闭 AirPlay 接收器")
            print("\n自动尝试端口 5001...")
            try:
                app.run(host='0.0.0.0', port=5001, debug=True)
            except OSError:
                print("❌ 端口 5001 也被占用，请手动指定可用端口")
        else:
            raise
