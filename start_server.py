#!/usr/bin/env python3
"""
电力潮流计算服务启动脚本
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import flask_cors
        import numpy
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def install_dependencies():
    """安装依赖"""
    print("正在安装依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("✗ 依赖安装失败")
        return False

def start_server(host="0.0.0.0", port=5000, debug=True):
    """启动服务器"""
    print(f"启动电力潮流计算服务...")
    print(f"服务地址: http://{host}:{port}")
    print(f"调试模式: {'开启' if debug else '关闭'}")
    print("-" * 50)
    
    # 设置环境变量
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development' if debug else 'production'
    
    # 导入并启动应用
    from app import app
    app.run(host=host, port=port, debug=debug)

def main():
    parser = argparse.ArgumentParser(description='电力潮流计算服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--no-debug', action='store_true', help='关闭调试模式')
    parser.add_argument('--install-deps', action='store_true', help='安装依赖')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        check_dependencies()
        return
    
    # 安装依赖
    if args.install_deps:
        install_dependencies()
        return
    
    # 检查依赖是否存在
    if not check_dependencies():
        print("\n是否要自动安装依赖? (y/n): ", end="")
        if input().lower() == 'y':
            if install_dependencies():
                print("请重新运行启动脚本")
            return
        else:
            print("请手动安装依赖后重新运行")
            return
    
    # 启动服务器
    start_server(
        host=args.host,
        port=args.port,
        debug=not args.no_debug
    )

if __name__ == '__main__':
    main()
