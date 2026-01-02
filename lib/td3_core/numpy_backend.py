#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯NumPy后端模块 - 深度学习基础组件
提供激活函数、初始化方法、优化器等核心功能
"""
import numpy as np
from typing import Dict, Tuple


# ========================== 全局配置 ==========================
DTYPE = np.float32  # 统一数据类型
EPS = 1e-8          # 数值稳定性epsilon
BETA1 = 0.9         # Adam优化器beta1
BETA2 = 0.999       # Adam优化器beta2
MAX_GRAD_NORM = 1.0 # 梯度裁剪最大范数


# ========================== 激活函数 ==========================
def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU激活函数
    
    Args:
        x: 输入数组
    
    Returns:
        激活后的数组
    """
    return np.maximum(0, x).astype(DTYPE)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    ReLU导数
    
    Args:
        x: 激活后的输出
    
    Returns:
        导数数组 (x > 0 时为1，否则为0)
    """
    return (x > 0).astype(DTYPE)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh激活函数
    
    Args:
        x: 输入数组
    
    Returns:
        激活后的数组
    """
    return np.tanh(x).astype(DTYPE)


def tanh_derivative(tanh_output: np.ndarray) -> np.ndarray:
    """
    Tanh导数
    
    Args:
        tanh_output: tanh函数的输出
    
    Returns:
        导数数组 (1 - tanh^2)
    """
    return (1 - np.square(tanh_output)).astype(DTYPE)


# ========================== 参数初始化 ==========================
def xavier_uniform(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """
    Xavier均匀分布初始化（对齐PyTorch的nn.init.xavier_uniform_）
    
    Args:
        shape: 参数形状 (fan_in, fan_out)
        gain: 增益系数
    
    Returns:
        初始化后的参数数组
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(DTYPE)


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """
    零初始化
    
    Args:
        shape: 参数形状
    
    Returns:
        全零数组
    """
    return np.zeros(shape, dtype=DTYPE)


# ========================== 梯度处理 ==========================
def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float = MAX_GRAD_NORM) -> Dict[str, np.ndarray]:
    """
    梯度裁剪（对齐PyTorch的torch.nn.utils.clip_grad_norm_）
    
    Args:
        grads: 梯度字典 {参数名: 梯度数组}
        max_norm: 最大梯度范数
    
    Returns:
        裁剪后的梯度字典
    """
    # 计算总L2范数
    total_norm = 0.0
    for g in grads.values():
        total_norm += np.sum(np.square(g))
    total_norm = np.sqrt(total_norm)
    
    # 计算裁剪系数
    clip_coef = max_norm / (total_norm + EPS)
    if clip_coef < 1.0:
        for k in grads.keys():
            grads[k] = grads[k] * clip_coef
    
    return grads


# ========================== 优化器 ==========================
class AdamOptimizer:
    """
    Adam优化器（完全对齐PyTorch的optim.Adam）
    
    实现细节：
    - 一阶动量：m_t = β1 * m_{t-1} + (1-β1) * g_t
    - 二阶动量：v_t = β2 * v_{t-1} + (1-β2) * g_t^2
    - 偏差修正：m_hat = m_t / (1 - β1^t), v_hat = v_t / (1 - β2^t)
    - 参数更新：θ_t = θ_{t-1} - lr * m_hat / (√v_hat + ε)
    """
    
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3, 
                 beta1: float = BETA1, beta2: float = BETA2, eps: float = EPS):
        """
        初始化Adam优化器
        
        Args:
            params: 参数字典 {参数名: 参数数组}
            lr: 学习率
            beta1: 一阶动量衰减系数
            beta2: 二阶动量衰减系数
            eps: 数值稳定性epsilon
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步
        
        # 初始化动量（严格匹配参数形状和类型）
        self.m = {k: np.zeros_like(v, dtype=DTYPE) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=DTYPE) for k, v in params.items()}
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行一步优化
        
        Args:
            params: 当前参数字典
            grads: 梯度字典
        
        Returns:
            更新后的参数字典
        """
        self.t += 1
        
        for k in params.keys():
            if k not in grads:
                raise KeyError(f"梯度字典缺少参数{k}，可用梯度：{list(grads.keys())}")
            
            grad = grads[k].astype(DTYPE)
            
            # 更新一阶动量
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            
            # 更新二阶动量
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * np.square(grad)
            
            # 偏差修正
            m_hat = self.m[k] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[k] / (1 - np.power(self.beta2, self.t))
            
            # 参数更新
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params
    
    def zero_grad(self):
        """清空梯度（保持接口一致性，NumPy版本不需要实际操作）"""
        pass


# ========================== 损失函数 ==========================
def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    均方误差损失（对齐PyTorch的nn.MSELoss）
    
    Args:
        predictions: 预测值
        targets: 目标值
    
    Returns:
        MSE损失值
    """
    return np.mean(np.square(predictions - targets))


def mse_loss_gradient(predictions: np.ndarray, targets: np.ndarray, batch_size: int) -> np.ndarray:
    """
    MSE损失的梯度
    
    Args:
        predictions: 预测值
        targets: 目标值
        batch_size: 批次大小
    
    Returns:
        损失对预测值的梯度
    """
    return 2 * (predictions - targets) / batch_size


# ========================== 工具函数 ==========================
def soft_update(target_params: Dict[str, np.ndarray], 
                source_params: Dict[str, np.ndarray], 
                tau: float = 0.005) -> Dict[str, np.ndarray]:
    """
    软更新目标网络参数（对齐PyTorch的软更新逻辑）
    
    target = τ * source + (1 - τ) * target
    
    Args:
        target_params: 目标网络参数
        source_params: 源网络参数
        tau: 软更新系数
    
    Returns:
        更新后的目标网络参数
    """
    updated_params = {}
    for k in target_params.keys():
        updated_params[k] = tau * source_params[k] + (1 - tau) * target_params[k]
    return updated_params


def copy_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    深拷贝参数字典
    
    Args:
        params: 参数字典
    
    Returns:
        拷贝后的参数字典
    """
    return {k: v.copy() for k, v in params.items()}


def concatenate_along_axis1(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    沿axis=1拼接数组（对齐PyTorch的torch.cat([x, y], dim=1)）
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
    
    Returns:
        拼接后的数组
    """
    return np.concatenate([arr1, arr2], axis=1).astype(DTYPE)


def clip_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    裁剪数组值到指定范围（对齐PyTorch的clamp）
    
    Args:
        arr: 输入数组
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        裁剪后的数组
    """
    return np.clip(arr, min_val, max_val).astype(DTYPE)

