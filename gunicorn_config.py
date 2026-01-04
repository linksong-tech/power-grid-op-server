import multiprocessing
import os

bind = "0.0.0.0:5002"
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
timeout = 300
keepalive = 5

max_requests = 1000
max_requests_jitter = 50

accesslog = "-"
errorlog = "-"
loglevel = "info"

preload_app = True

def on_starting(server):
    print("=" * 60)
    print("电力潮流计算服务启动中...")
    print(f"服务地址: http://0.0.0.0:5002")
    print(f"Workers: {workers}")
    print(f"Worker Class: {worker_class}")
    print("=" * 60)

def worker_int(worker):
    print(f"Worker {worker.pid} 收到中断信号")

def worker_abort(worker):
    print(f"Worker {worker.pid} 异常终止")

