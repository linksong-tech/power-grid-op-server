#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3服务使用示例
演示如何使用工程化的训练和推理API
"""
import numpy as np
from td3_train_service_numpy import train_td3_model
from td3_inference_service_numpy import optimize_reactive_power, batch_optimize


# ==================== 示例1: 训练TD3模型 ====================
def example_train():
    """训练示例"""
    print("=== 示例1: 训练TD3模型 ===\n")
    
    # 准备训练数据
    bus_data = np.array([
        [1, -5.0, 0.0],    # 平衡节点
        [2, -2.0, -1.0],   # 光伏节点
        [3, 1.5, 0.8],     # 负荷节点
        [4, 1.2, 0.6],     # 负荷节点
        [5, 0.8, 0.4],     # 负荷节点
    ])
    
    branch_data = np.array([
        [1, 1, 2, 0.05, 0.15],
        [2, 2, 3, 0.08, 0.20],
        [3, 2, 4, 0.06, 0.18],
        [4, 3, 5, 0.04, 0.12],
    ])
    
    voltage_limits = (9.5, 10.5)  # kV
    key_nodes = [1, 2, 3]  # 关键监测节点索引
    tunable_q_nodes = [
        (1, -2.0, 2.0, "光伏节点2"),  # (节点索引, Q_min, Q_max, 节点名)
    ]
    ub = 10.0  # 基准电压
    
    # 进度回调函数
    def progress_callback(episode, reward, loss_rate):
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={reward:.2f}, Loss Rate={loss_rate:.4f}%")
    
    # 执行训练
    result = train_td3_model(
        bus_data=bus_data,
        branch_data=branch_data,
        voltage_limits=voltage_limits,
        key_nodes=key_nodes,
        tunable_q_nodes=tunable_q_nodes,
        ub=ub,
        max_episodes=500,
        max_steps=3,
        model_save_path="trained_model.pth",
        progress_callback=progress_callback
    )
    
    print(f"\n训练完成!")
    print(f"成功: {result['success']}")
    print(f"最优网损率: {result['best_loss_rate']:.4f}%")
    print(f"模型保存路径: {result['model_path']}")
    
    return result


# ==================== 示例2: 单次优化 ====================
def example_single_optimization():
    """单次优化示例"""
    print("\n=== 示例2: 单次无功优化 ===\n")
    
    # 准备数据（与训练时相同的网络结构）
    bus_data = np.array([
        [1, -5.2, 0.0],
        [2, -2.1, -0.8],
        [3, 1.6, 0.9],
        [4, 1.3, 0.7],
        [5, 0.9, 0.5],
    ])
    
    branch_data = np.array([
        [1, 1, 2, 0.05, 0.15],
        [2, 2, 3, 0.08, 0.20],
        [3, 2, 4, 0.06, 0.18],
        [4, 3, 5, 0.04, 0.12],
    ])
    
    voltage_limits = (9.5, 10.5)
    key_nodes = [1, 2, 3]
    tunable_q_nodes = [(1, -2.0, 2.0, "光伏节点2")]
    ub = 10.0
    model_path = "trained_model_1229_230322.pth"  # 训练好的模型
    
    # 执行优化
    result = optimize_reactive_power(
        bus_data=bus_data,
        branch_data=branch_data,
        voltage_limits=voltage_limits,
        key_nodes=key_nodes,
        tunable_q_nodes=tunable_q_nodes,
        model_path=model_path,
        ub=ub
    )
    
    if result['success']:
        print("优化成功!")
        print(f"优化前网损率: {result['initial_loss_rate']:.4f}%")
        print(f"优化后网损率: {result['optimized_loss_rate']:.4f}%")
        print(f"网损降低: {result['loss_reduction']:.2f}%")
        print(f"电压越限节点数: {result['voltage_violations']}")
        print("\n无功调节详情:")
        for adj in result['q_adjustments']:
            print(f"  {adj['node_name']}: {adj['initial_q']} → {adj['optimized_q']} MVar "
                  f"(调整: {adj['adjustment']:+.4f})")
    else:
        print(f"优化失败: {result['error']}")
    
    return result


# ==================== 示例3: 批量优化 ====================
def example_batch_optimization():
    """批量优化示例"""
    print("\n=== 示例3: 批量优化多个断面 ===\n")
    
    # 准备多个测试样本
    test_samples = [
        {
            'time': '2024-12-29 08:00',
            'ub': 10.0,
            'bus': np.array([1, -5.0, 0.0, 2, -2.0, -1.0, 3, 1.5, 0.8, 4, 1.2, 0.6, 5, 0.8, 0.4])
        },
        {
            'time': '2024-12-29 12:00',
            'ub': 10.0,
            'bus': np.array([1, -6.0, 0.0, 2, -2.5, -1.2, 3, 1.8, 1.0, 4, 1.5, 0.8, 5, 1.0, 0.5])
        },
        {
            'time': '2024-12-29 18:00',
            'ub': 10.0,
            'bus': np.array([1, -4.5, 0.0, 2, -1.8, -0.9, 3, 1.3, 0.7, 4, 1.0, 0.5, 5, 0.7, 0.3])
        },
    ]
    
    branch_data = np.array([
        [1, 1, 2, 0.05, 0.15],
        [2, 2, 3, 0.08, 0.20],
        [3, 2, 4, 0.06, 0.18],
        [4, 3, 5, 0.04, 0.12],
    ])
    
    voltage_limits = (9.5, 10.5)
    key_nodes = [1, 2, 3]
    tunable_q_nodes = [(1, -2.0, 2.0, "光伏节点2")]
    model_path = "trained_model_1229_230322.pth"
    
    # 执行批量优化
    result = batch_optimize(
        test_samples=test_samples,
        branch_data=branch_data,
        voltage_limits=voltage_limits,
        key_nodes=key_nodes,
        tunable_q_nodes=tunable_q_nodes,
        model_path=model_path
    )
    
    print(f"批量优化完成!")
    print(f"总样本数: {result['total_samples']}")
    print(f"成功优化: {result['successful_optimizations']}")
    
    if result['summary']:
        print(f"\n汇总统计:")
        print(f"  平均优化前网损率: {result['summary']['avg_initial_loss']:.4f}%")
        print(f"  平均优化后网损率: {result['summary']['avg_optimized_loss']:.4f}%")
        print(f"  平均网损降低: {result['summary']['avg_loss_reduction']:.2f}%")
        print(f"  平均电压越限节点数: {result['summary']['avg_voltage_violations']:.2f}")
    
    print("\n各断面详情:")
    for r in result['results']:
        if r['success']:
            print(f"  {r['sample_time']}: {r['initial_loss_rate']:.4f}% → "
                  f"{r['optimized_loss_rate']:.4f}% (降低{r['loss_reduction']:.2f}%)")
        else:
            print(f"  {r['sample_time']}: 失败 - {r['error']}")
    
    return result


# ==================== 示例4: 与PSO对比 ====================
def example_compare_with_pso():
    """与PSO算法对比示例"""
    print("\n=== 示例4: TD3 vs PSO性能对比 ===\n")
    
    # 这里可以调用PSO优化函数进行对比
    # 从 pso_optimize.py 导入相关函数
    
    print("提示: 可以使用 td3use.py 中的 batch_validate_model 函数")
    print("该函数会自动对比TD3和PSO的性能，并生成详细报告")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("TD3服务使用示例\n")
    print("=" * 60)
    
    # 运行示例（根据需要取消注释）
    
    # 示例1: 训练模型
    # example_train()
    
    # 示例2: 单次优化
    example_single_optimization()
    
    # 示例3: 批量优化
    # example_batch_optimization()
    
    # 示例4: 与PSO对比
    # example_compare_with_pso()
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
