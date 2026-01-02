#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯NumPy实现的TD3神经网络模型
完全替代PyTorch版本，提供相同的接口
"""
import numpy as np
from typing import Dict, Tuple
from .numpy_backend import (
    relu, relu_derivative, tanh, tanh_derivative,
    xavier_uniform, zeros, DTYPE
)


class ActorNetworkNumpy:
    """
    Actor网络（策略网络）- 纯NumPy实现
    
    网络结构：
    - Input: state_dim
    - Hidden1: 128 (ReLU)
    - Hidden2: 64 (ReLU)
    - Hidden3: 32 (ReLU)
    - Output: action_dim (Tanh)
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 最大动作值
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 初始化网络参数（对齐PyTorch的Sequential结构）
        self.params = {
            "w1": xavier_uniform((state_dim, 128), gain=1.0),
            "b1": zeros((128,)),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": zeros((64,)),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": zeros((32,)),
            "w4": xavier_uniform((32, action_dim), gain=1.0),
            "b4": zeros((action_dim,))
        }
        
        # 用于存储前向传播的中间结果（反向传播需要）
        self.cache = {}
    
    def forward(self, state: np.ndarray, save_cache: bool = False) -> np.ndarray:
        """
        前向传播
        
        Args:
            state: 状态输入 (batch_size, state_dim)
            save_cache: 是否保存中间结果（训练时需要）
        
        Returns:
            动作输出 (batch_size, action_dim)
        """
        state = state.astype(DTYPE)
        
        # Layer 1: Linear + ReLU
        z1 = np.dot(state, self.params["w1"]) + self.params["b1"]
        h1 = relu(z1)
        
        # Layer 2: Linear + ReLU
        z2 = np.dot(h1, self.params["w2"]) + self.params["b2"]
        h2 = relu(z2)
        
        # Layer 3: Linear + ReLU
        z3 = np.dot(h2, self.params["w3"]) + self.params["b3"]
        h3 = relu(z3)
        
        # Layer 4: Linear + Tanh
        z4 = np.dot(h3, self.params["w4"]) + self.params["b4"]
        h4 = tanh(z4)
        
        # 输出缩放
        output = self.max_action * h4
        
        # 保存中间结果（用于反向传播）
        if save_cache:
            self.cache = {
                "input": state,
                "z1": z1, "h1": h1,
                "z2": z2, "h2": h2,
                "z3": z3, "h3": h3,
                "z4": z4, "h4": h4,
                "output": output
            }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        反向传播计算梯度
        
        Args:
            grad_output: 损失对输出的梯度 (batch_size, action_dim)
        
        Returns:
            参数梯度字典
        """
        if not self.cache:
            raise RuntimeError("必须先调用forward(save_cache=True)才能进行反向传播")
        
        batch_size = self.cache["input"].shape[0]
        
        # 输出层梯度（考虑max_action缩放）
        grad_h4 = grad_output * self.max_action
        
        # Layer 4: Tanh导数
        grad_z4 = grad_h4 * tanh_derivative(self.cache["h4"])
        grad_w4 = np.dot(self.cache["h3"].T, grad_z4)
        grad_b4 = np.sum(grad_z4, axis=0)
        
        # Layer 3: ReLU导数
        grad_h3 = np.dot(grad_z4, self.params["w4"].T)
        grad_z3 = grad_h3 * relu_derivative(self.cache["h3"])
        grad_w3 = np.dot(self.cache["h2"].T, grad_z3)
        grad_b3 = np.sum(grad_z3, axis=0)
        
        # Layer 2: ReLU导数
        grad_h2 = np.dot(grad_z3, self.params["w3"].T)
        grad_z2 = grad_h2 * relu_derivative(self.cache["h2"])
        grad_w2 = np.dot(self.cache["h1"].T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)
        
        # Layer 1: ReLU导数
        grad_h1 = np.dot(grad_z2, self.params["w2"].T)
        grad_z1 = grad_h1 * relu_derivative(self.cache["h1"])
        grad_w1 = np.dot(self.cache["input"].T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)
        
        return {
            "w1": grad_w1, "b1": grad_b1,
            "w2": grad_w2, "b2": grad_b2,
            "w3": grad_w3, "b3": grad_b3,
            "w4": grad_w4, "b4": grad_b4
        }
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """获取参数副本"""
        return {k: v.copy() for k, v in self.params.items()}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """设置参数"""
        for k, v in params.items():
            self.params[k] = v.astype(DTYPE)
    
    def save(self, filepath: str):
        """保存模型参数到.npz文件"""
        np.savez(filepath, **{f"actor_{k}": v for k, v in self.params.items()})
    
    def load(self, filepath: str):
        """从.npz文件加载模型参数"""
        data = np.load(filepath)
        for k in self.params.keys():
            npz_key = f"actor_{k}"
            if npz_key in data:
                self.params[k] = data[npz_key].astype(DTYPE)
        data.close()


class CriticNetworkNumpy:
    """
    Critic网络（价值网络，双Q网络）- 纯NumPy实现
    
    网络结构（两个独立的Q网络）：
    - Input: state_dim + action_dim
    - Hidden1: 128 (ReLU)
    - Hidden2: 64 (ReLU)
    - Hidden3: 32 (ReLU)
    - Output: 1
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        
        # 网络1参数
        self.params1 = {
            "w1": xavier_uniform((self.input_dim, 128), gain=1.0),
            "b1": zeros((128,)),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": zeros((64,)),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": zeros((32,)),
            "w4": xavier_uniform((32, 1), gain=1.0),
            "b4": zeros((1,))
        }
        
        # 网络2参数
        self.params2 = {
            "w1": xavier_uniform((self.input_dim, 128), gain=1.0),
            "b1": zeros((128,)),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": zeros((64,)),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": zeros((32,)),
            "w4": xavier_uniform((32, 1), gain=1.0),
            "b4": zeros((1,))
        }
        
        # 用于存储前向传播的中间结果
        self.cache1 = {}
        self.cache2 = {}
    
    def _forward_single(self, x: np.ndarray, params: Dict[str, np.ndarray], 
                        save_cache: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        单个Q网络的前向传播
        
        Args:
            x: 输入 (batch_size, state_dim + action_dim)
            params: 网络参数
            save_cache: 是否保存中间结果
        
        Returns:
            Q值输出和缓存字典
        """
        x = x.astype(DTYPE)
        
        # Layer 1
        z1 = np.dot(x, params["w1"]) + params["b1"]
        h1 = relu(z1)
        
        # Layer 2
        z2 = np.dot(h1, params["w2"]) + params["b2"]
        h2 = relu(z2)
        
        # Layer 3
        z3 = np.dot(h2, params["w3"]) + params["b3"]
        h3 = relu(z3)
        
        # Layer 4 (输出层，无激活)
        q = np.dot(h3, params["w4"]) + params["b4"]
        
        cache = {}
        if save_cache:
            cache = {
                "input": x,
                "z1": z1, "h1": h1,
                "z2": z2, "h2": h2,
                "z3": z3, "h3": h3,
                "q": q
            }
        
        return q, cache
    
    def forward(self, state: np.ndarray, action: np.ndarray, 
                save_cache: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播（返回两个Q值）
        
        Args:
            state: 状态输入 (batch_size, state_dim)
            action: 动作输入 (batch_size, action_dim)
            save_cache: 是否保存中间结果
        
        Returns:
            (Q1值, Q2值)
        """
        state = state.astype(DTYPE)
        action = action.astype(DTYPE)
        x = np.concatenate([state, action], axis=1)
        
        q1, cache1 = self._forward_single(x, self.params1, save_cache)
        q2, cache2 = self._forward_single(x, self.params2, save_cache)
        
        if save_cache:
            self.cache1 = cache1
            self.cache2 = cache2
        
        return q1, q2
    
    def Q1(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """仅返回第一个Q网络的输出"""
        state = state.astype(DTYPE)
        action = action.astype(DTYPE)
        x = np.concatenate([state, action], axis=1)
        q1, _ = self._forward_single(x, self.params1, save_cache=False)
        return q1
    
    def _backward_single(self, grad_q: np.ndarray, cache: Dict, 
                         params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        单个Q网络的反向传播
        
        Args:
            grad_q: 损失对Q值的梯度
            cache: 前向传播缓存
            params: 网络参数
        
        Returns:
            参数梯度字典
        """
        # Layer 4梯度
        grad_w4 = np.dot(cache["h3"].T, grad_q)
        grad_b4 = np.sum(grad_q, axis=0)
        
        # Layer 3梯度
        grad_h3 = np.dot(grad_q, params["w4"].T)
        grad_z3 = grad_h3 * relu_derivative(cache["h3"])
        grad_w3 = np.dot(cache["h2"].T, grad_z3)
        grad_b3 = np.sum(grad_z3, axis=0)
        
        # Layer 2梯度
        grad_h2 = np.dot(grad_z3, params["w3"].T)
        grad_z2 = grad_h2 * relu_derivative(cache["h2"])
        grad_w2 = np.dot(cache["h1"].T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)
        
        # Layer 1梯度
        grad_h1 = np.dot(grad_z2, params["w2"].T)
        grad_z1 = grad_h1 * relu_derivative(cache["h1"])
        grad_w1 = np.dot(cache["input"].T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)
        
        return {
            "w1": grad_w1, "b1": grad_b1,
            "w2": grad_w2, "b2": grad_b2,
            "w3": grad_w3, "b3": grad_b3,
            "w4": grad_w4, "b4": grad_b4
        }
    
    def backward(self, grad_q1: np.ndarray, grad_q2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        反向传播计算两个网络的梯度
        
        Args:
            grad_q1: 损失对Q1的梯度
            grad_q2: 损失对Q2的梯度
        
        Returns:
            (网络1梯度字典, 网络2梯度字典)
        """
        if not self.cache1 or not self.cache2:
            raise RuntimeError("必须先调用forward(save_cache=True)才能进行反向传播")
        
        grads1 = self._backward_single(grad_q1, self.cache1, self.params1)
        grads2 = self._backward_single(grad_q2, self.cache2, self.params2)
        
        return grads1, grads2
    
    def get_params(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """获取两个网络的参数副本"""
        return (
            {k: v.copy() for k, v in self.params1.items()},
            {k: v.copy() for k, v in self.params2.items()}
        )
    
    def set_params(self, params1: Dict[str, np.ndarray], params2: Dict[str, np.ndarray]):
        """设置网络参数"""
        for k, v in params1.items():
            self.params1[k] = v.astype(DTYPE)
        for k, v in params2.items():
            self.params2[k] = v.astype(DTYPE)
    
    def save(self, filepath: str):
        """保存模型参数到.npz文件"""
        save_dict = {}
        save_dict.update({f"critic1_{k}": v for k, v in self.params1.items()})
        save_dict.update({f"critic2_{k}": v for k, v in self.params2.items()})
        np.savez(filepath, **save_dict)
    
    def load(self, filepath: str):
        """从.npz文件加载模型参数"""
        data = np.load(filepath)
        for k in self.params1.keys():
            npz_key = f"critic1_{k}"
            if npz_key in data:
                self.params1[k] = data[npz_key].astype(DTYPE)
        for k in self.params2.keys():
            npz_key = f"critic2_{k}"
            if npz_key in data:
                self.params2[k] = data[npz_key].astype(DTYPE)
        data.close()

