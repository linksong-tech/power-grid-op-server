#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3推理服务 - 纯NumPy版本
完全替代PyTorch实现，提供相同的推理接口
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

from .td3_core.numpy_models import ActorNetworkNumpy
from .td3_core.power_flow import power_flow_calculation
from .td3_core.numpy_backend import DTYPE


class TD3InferenceServiceNumpy:
    """TD3推理服务类 - 纯NumPy实现"""
    
    def __init__(
        self,
        model_path: str,
        state_dim: int,
        action_dim: int,
        voltage_limits: Tuple[float, float],
        key_nodes: List[int],
        max_action: float = 1.0
    ):
        """
        初始化TD3推理服务
        
        Args:
            model_path: 训练好的模型文件路径（.npz格式）
            state_dim: 状态维度（关键节点数量）
            action_dim: 动作维度（可调无功节点数量）
            voltage_limits: 电压上下限 (v_min, v_max)
            key_nodes: 关键节点索引列表
            max_action: 最大动作值
        """
        print(f"使用设备: CPU (NumPy {DTYPE.__name__})")
        
        # 加载模型
        self.actor = ActorNetworkNumpy(state_dim, action_dim, max_action)
        
        try:
            model_data = np.load(model_path)
            # 加载Actor参数
            for k in self.actor.params.keys():
                npz_key = f"actor_{k}"
                if npz_key in model_data:
                    self.actor.params[k] = model_data[npz_key].astype(DTYPE)
                else:
                    raise KeyError(f"模型文件缺少参数: {npz_key}")
            model_data.close()
            print(f"成功加载模型：{model_path}")
        except Exception as e:
            raise ValueError(f"模型加载失败：{e}\n请检查模型文件路径和完整性！")
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.v_min, self.v_max = voltage_limits
        self.key_nodes = key_nodes
    
    def _build_state(self, observed_voltages: np.ndarray) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            observed_voltages: 观测到的节点电压 (kV)
        
        Returns:
            归一化的状态向量
        """
        # 提取关键节点电压
        key_node_voltages = observed_voltages[self.key_nodes]
        
        # 电压归一化到[-1, 1]
        normalized_voltages = (key_node_voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        normalized_voltages = np.clip(normalized_voltages, -1, 1)
        
        return normalized_voltages.astype(DTYPE)
    
    def _denormalize_action(
        self, 
        action: np.ndarray, 
        tunable_q_nodes: List[Tuple[int, float, float, str]]
    ) -> List[float]:
        """
        动作反归一化
        
        Args:
            action: 归一化的动作 [-1, 1]
            tunable_q_nodes: 可调无功节点配置
        
        Returns:
            实际无功值列表 (MVar)
        """
        q_mins = np.array([node[1] for node in tunable_q_nodes])
        q_maxs = np.array([node[2] for node in tunable_q_nodes])
        
        actual_actions = []
        for i in range(len(action)):
            normalized = np.clip(action[i], -1, 1)
            actual = (normalized + 1) / 2 * (q_maxs[i] - q_mins[i]) + q_mins[i]
            actual_actions.append(actual)
        return actual_actions
    
    def predict(
        self, 
        observed_voltages: np.ndarray,
        tunable_q_nodes: List[Tuple[int, float, float, str]]
    ) -> List[float]:
        """
        预测最优无功配置
        
        Args:
            observed_voltages: 观测到的节点电压 (kV)
            tunable_q_nodes: 可调无功节点配置
        
        Returns:
            最优无功值列表 (MVar)
        """
        # 构建状态
        state = self._build_state(observed_voltages)
        
        # 转换为模型输入格式 (1, state_dim)
        state_input = state.reshape(1, -1)
        
        # 模型前向传播（纯推理，不保存缓存）
        normalized_action = self.actor.forward(state_input, save_cache=False).flatten()
        
        # 反归一化
        actual_action = self._denormalize_action(normalized_action, tunable_q_nodes)
        
        return actual_action


def optimize_reactive_power(
    bus_data: np.ndarray,
    branch_data: np.ndarray,
    voltage_limits: Tuple[float, float],
    key_nodes: List[int],
    tunable_q_nodes: List[Tuple[int, float, float, str]],
    model_path: str,
    ub: float,
    sb: float = 10.0,
    pr: float = 1e-6
) -> Dict:
    """
    使用TD3模型优化无功功率配置（纯NumPy版本）
    
    Args:
        bus_data: 节点数据 (n×3) [节点号, 有功P, 无功Q]
        branch_data: 支路数据 (m×5) [线路号, 首节点, 末节点, 电阻R, 电抗X]
        voltage_limits: 电压上下限 (v_min, v_max) kV
        key_nodes: 关键节点索引列表
        tunable_q_nodes: 可调无功节点配置 [(节点索引, Q_min, Q_max, 节点名), ...]
        model_path: 训练好的模型文件路径（.npz格式）
        ub: 基准电压 kV
        sb: 基准功率 MVA
        pr: 潮流收敛精度
    
    Returns:
        dict: {
            'success': bool,
            'optimized_q_values': List[float],  # 优化后的无功值
            'initial_loss_rate': float,  # 优化前网损率
            'optimized_loss_rate': float,  # 优化后网损率
            'loss_reduction': float,  # 网损降低百分比
            'optimized_voltages': np.ndarray,  # 优化后节点电压
            'voltage_violations': int,  # 电压越限节点数
            'q_adjustments': List[Dict]  # 无功调节详情
        }
    """
    try:
        # 1. 计算优化前状态
        initial_q = [bus_data[node[0], 2] for node in tunable_q_nodes]
        initial_loss, initial_voltages, _ = power_flow_calculation(
            bus_data, branch_data, tunable_q_nodes, initial_q, sb, ub, pr
        )
        
        if initial_loss is None:
            return {
                'success': False,
                'error': '初始潮流计算未收敛'
            }
        
        # 2. 初始化推理服务
        state_dim = len(key_nodes)
        action_dim = len(tunable_q_nodes)
        
        inference_service = TD3InferenceServiceNumpy(
            model_path=model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes
        )
        
        # 3. 预测最优无功配置
        optimized_q = inference_service.predict(initial_voltages, tunable_q_nodes)
        
        # 4. 计算优化后状态
        optimized_loss, optimized_voltages, _ = power_flow_calculation(
            bus_data, branch_data, tunable_q_nodes, optimized_q, sb, ub, pr
        )
        
        if optimized_loss is None:
            return {
                'success': False,
                'error': '优化后潮流计算未收敛'
            }
        
        # 5. 计算电压越限
        v_min, v_max = voltage_limits
        voltage_violations = sum(1 for v in optimized_voltages if v < v_min or v > v_max)
        
        # 6. 构建无功调节详情
        q_adjustments = []
        for i, (node_idx, q_min, q_max, node_name) in enumerate(tunable_q_nodes):
            q_adjustments.append({
                'node_name': node_name,
                'node_index': node_idx,
                'initial_q': round(initial_q[i], 4),
                'optimized_q': round(optimized_q[i], 4),
                'adjustment': round(optimized_q[i] - initial_q[i], 4),
                'q_limits': [q_min, q_max]
            })
        
        # 7. 返回结果
        loss_reduction = ((initial_loss - optimized_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        return {
            'success': True,
            'optimized_q_values': [round(q, 4) for q in optimized_q],
            'initial_loss_rate': round(initial_loss, 4),
            'optimized_loss_rate': round(optimized_loss, 4),
            'loss_reduction': round(loss_reduction, 2),
            'optimized_voltages': optimized_voltages,
            'voltage_violations': voltage_violations,
            'q_adjustments': q_adjustments
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def batch_optimize(
    test_samples: List[Dict],
    branch_data: np.ndarray,
    voltage_limits: Tuple[float, float],
    key_nodes: List[int],
    tunable_q_nodes: List[Tuple[int, float, float, str]],
    model_path: str,
    sb: float = 10.0,
    pr: float = 1e-6
) -> Dict:
    """
    批量优化多个测试样本（纯NumPy版本）
    
    Args:
        test_samples: 测试样本列表，每个样本格式: {
            'time': str,
            'ub': float,
            'bus': np.ndarray
        }
        branch_data: 支路数据
        voltage_limits: 电压上下限
        key_nodes: 关键节点索引
        tunable_q_nodes: 可调无功节点配置
        model_path: 模型路径（.npz格式）
        sb: 基准功率
        pr: 潮流收敛精度
    
    Returns:
        dict: {
            'success': bool,
            'total_samples': int,
            'successful_optimizations': int,
            'results': List[Dict],
            'summary': {
                'avg_initial_loss': float,
                'avg_optimized_loss': float,
                'avg_loss_reduction': float,
                'avg_voltage_violations': float
            }
        }
    """
    results = []
    successful_count = 0
    
    print(f"\n=== 开始批量优化（共{len(test_samples)}个样本）===")
    
    for idx, sample in enumerate(test_samples):
        sample_time = sample['time']
        ub = sample['ub']
        bus_data = sample['bus'].reshape(-1, 3)
        
        print(f"\n处理样本 {idx+1}/{len(test_samples)}: {sample_time}")
        
        result = optimize_reactive_power(
            bus_data=bus_data,
            branch_data=branch_data,
            voltage_limits=voltage_limits,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_q_nodes,
            model_path=model_path,
            ub=ub,
            sb=sb,
            pr=pr
        )
        
        result['sample_time'] = sample_time
        results.append(result)
        
        if result['success']:
            successful_count += 1
            print(f"✓ 优化成功 - 网损率: {result['initial_loss_rate']:.4f}% → {result['optimized_loss_rate']:.4f}% (降低{result['loss_reduction']:.2f}%)")
        else:
            print(f"✗ 优化失败 - {result.get('error', '未知错误')}")
    
    # 计算汇总统计
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        summary = {
            'avg_initial_loss': round(np.mean([r['initial_loss_rate'] for r in successful_results]), 4),
            'avg_optimized_loss': round(np.mean([r['optimized_loss_rate'] for r in successful_results]), 4),
            'avg_loss_reduction': round(np.mean([r['loss_reduction'] for r in successful_results]), 2),
            'avg_voltage_violations': round(np.mean([r['voltage_violations'] for r in successful_results]), 2)
        }
        
        print(f"\n=== 批量优化完成 ===")
        print(f"成功样本数: {successful_count}/{len(test_samples)}")
        print(f"平均优化前网损率: {summary['avg_initial_loss']:.4f}%")
        print(f"平均优化后网损率: {summary['avg_optimized_loss']:.4f}%")
        print(f"平均网损降低: {summary['avg_loss_reduction']:.2f}%")
        print(f"平均电压越限节点数: {summary['avg_voltage_violations']:.2f}")
    else:
        summary = {}
        print(f"\n=== 批量优化完成 ===")
        print(f"所有样本优化失败！")
    
    return {
        'success': True,
        'total_samples': len(test_samples),
        'successful_optimizations': successful_count,
        'results': results,
        'summary': summary
    }

