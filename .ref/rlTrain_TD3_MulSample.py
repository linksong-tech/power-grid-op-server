#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 23:51:43 2026

@author: ryne
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用化的电力系统无功优化TD3强化学习训练程序（纯NumPy版本，修复维度不匹配）
支持从指定Excel文件读取配置参数和训练数据
支持多样本分层训练（每个样本连续训练多个episode）
自动计算基准网损率并进行归一化训练
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import random
import datetime
import os
import glob
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import deque  # 对齐PyTorch的经验回放容器

# -------------------------- 全局配置（对齐PyTorch） --------------------------
DTYPE = np.float32  # 统一数据类型为float32（PyTorch默认）
EPS = 1e-8          # Adam优化器epsilon（对齐PyTorch）
BETA1 = 0.9         # Adam beta1
BETA2 = 0.999       # Adam beta2
MAX_GRAD_NORM = 1.0 # 梯度裁剪最大范数

# -------------------------- 基础工具函数（对齐PyTorch） --------------------------
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数（对齐PyTorch的nn.ReLU）"""
    return np.maximum(0, x).astype(DTYPE)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数（对齐PyTorch的nn.Tanh）"""
    return np.tanh(x).astype(DTYPE)

def xavier_uniform(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """严格对齐PyTorch的nn.init.xavier_uniform_初始化"""
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(DTYPE)

def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float = MAX_GRAD_NORM) -> Dict[str, np.ndarray]:
    """严格对齐PyTorch的torch.nn.utils.clip_grad_norm_"""
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

# -------------------------- Adam优化器（严格对齐PyTorch） --------------------------
class AdamOptimizer:
    """纯NumPy实现的Adam优化器（完全对齐PyTorch的optim.Adam）"""
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3):
        self.lr = lr
        self.beta1 = BETA1
        self.beta2 = BETA2
        self.eps = EPS
        self.t = 0
        
        # 初始化动量和二阶动量（严格匹配参数形状和类型）
        self.m = {k: np.zeros_like(v, dtype=DTYPE) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=DTYPE) for k, v in params.items()}
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步优化（完全对齐PyTorch的Adam更新逻辑）"""
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
            
            # 参数更新（对齐PyTorch的更新公式）
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params

# -------------------------- 网络定义（对齐PyTorch） --------------------------
class ActorNetwork:
    """纯NumPy实现的Actor网络（完全对齐PyTorch版本）"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 严格对齐PyTorch的nn.Sequential初始化
        self.params = {
            "w1": xavier_uniform((state_dim, 128), gain=1.0),
            "b1": np.zeros(128, dtype=DTYPE),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": np.zeros(64, dtype=DTYPE),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": np.zeros(32, dtype=DTYPE),
            "w4": xavier_uniform((32, action_dim), gain=1.0),
            "b4": np.zeros(action_dim, dtype=DTYPE)
        }
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播（严格对齐PyTorch的前向逻辑）"""
        state = state.astype(DTYPE)
        x = relu(np.dot(state, self.params["w1"]) + self.params["b1"])
        x = relu(np.dot(x, self.params["w2"]) + self.params["b2"])
        x = relu(np.dot(x, self.params["w3"]) + self.params["b3"])
        x = tanh(np.dot(x, self.params["w4"]) + self.params["b4"])
        return self.max_action * x
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """获取参数副本（对齐PyTorch的state_dict）"""
        return {k: v.copy() for k, v in self.params.items()}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """设置参数（对齐PyTorch的load_state_dict）"""
        for k, v in params.items():
            self.params[k] = v.astype(DTYPE)

class CriticNetwork:
    """纯NumPy实现的双Critic网络（完全对齐PyTorch版本）"""
    def __init__(self, state_dim: int, action_dim: int):
        self.input_dim = state_dim + action_dim
        
        # 网络1参数（对齐PyTorch的network1）
        self.params1 = {
            "w1": xavier_uniform((self.input_dim, 128), gain=1.0),
            "b1": np.zeros(128, dtype=DTYPE),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": np.zeros(64, dtype=DTYPE),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": np.zeros(32, dtype=DTYPE),
            "w4": xavier_uniform((32, 1), gain=1.0),
            "b4": np.zeros(1, dtype=DTYPE)
        }
        
        # 网络2参数（对齐PyTorch的network2）
        self.params2 = {
            "w1": xavier_uniform((self.input_dim, 128), gain=1.0),
            "b1": np.zeros(128, dtype=DTYPE),
            "w2": xavier_uniform((128, 64), gain=1.0),
            "b2": np.zeros(64, dtype=DTYPE),
            "w3": xavier_uniform((64, 32), gain=1.0),
            "b3": np.zeros(32, dtype=DTYPE),
            "w4": xavier_uniform((32, 1), gain=1.0),
            "b4": np.zeros(1, dtype=DTYPE)
        }
    
    def forward(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播（返回两个Q值，对齐PyTorch）"""
        state = state.astype(DTYPE)
        action = action.astype(DTYPE)
        x = np.concatenate([state, action], axis=1)
        
        # 网络1前向
        q1 = relu(np.dot(x, self.params1["w1"]) + self.params1["b1"])
        q1 = relu(np.dot(q1, self.params1["w2"]) + self.params1["b2"])
        q1 = relu(np.dot(q1, self.params1["w3"]) + self.params1["b3"])
        q1 = np.dot(q1, self.params1["w4"]) + self.params1["b4"]
        
        # 网络2前向
        q2 = relu(np.dot(x, self.params2["w1"]) + self.params2["b1"])
        q2 = relu(np.dot(q2, self.params2["w2"]) + self.params2["b2"])
        q2 = relu(np.dot(q2, self.params2["w3"]) + self.params2["b3"])
        q2 = np.dot(q2, self.params2["w4"]) + self.params2["b4"]
        
        return q1, q2
    
    def Q1(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """仅返回第一个Q网络输出（对齐PyTorch）"""
        state = state.astype(DTYPE)
        action = action.astype(DTYPE)
        x = np.concatenate([state, action], axis=1)
        q1 = relu(np.dot(x, self.params1["w1"]) + self.params1["b1"])
        q1 = relu(np.dot(q1, self.params1["w2"]) + self.params1["b2"])
        q1 = relu(np.dot(q1, self.params1["w3"]) + self.params1["b3"])
        q1 = np.dot(q1, self.params1["w4"]) + self.params1["b4"]
        return q1
    
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

# -------------------------- TD3算法（完全对齐PyTorch版本） --------------------------
class TD3:
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        print(f"使用设备: CPU (NumPy float32)")
        
        # 初始化主网络
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim, action_dim)
        
        # 初始化目标网络并复制参数（对齐PyTorch的load_state_dict）
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action)
        self.actor_target.set_params(self.actor.get_params())
        
        self.critic_target = CriticNetwork(state_dim, action_dim)
        critic_params1, critic_params2 = self.critic.get_params()
        self.critic_target.set_params(critic_params1, critic_params2)
        
        # 初始化优化器（对齐PyTorch的学习率）
        self.actor_optimizer = AdamOptimizer(self.actor.get_params(), lr=1e-4)
        # 扁平化Critic参数用于优化器
        critic_flat_params = self._flatten_critic_params()
        self.critic_optimizer = AdamOptimizer(critic_flat_params, lr=1e-3)
        
        # 超参数（完全对齐PyTorch版本）
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=50000)  # 改用deque（对齐PyTorch）
        self.batch_size = 128
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
    
    def _flatten_critic_params(self) -> Dict[str, np.ndarray]:
        """扁平化Critic参数（用于优化器）"""
        params1, params2 = self.critic.get_params()
        flat_params = {}
        # 网络1参数
        for k, v in params1.items():
            flat_params[f"c1_{k}"] = v
        # 网络2参数
        for k, v in params2.items():
            flat_params[f"c2_{k}"] = v
        return flat_params
    
    def _unflatten_critic_params(self, flat_params: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """解扁平化Critic参数"""
        params1 = {}
        params2 = {}
        for k, v in flat_params.items():
            if k.startswith("c1_"):
                params1[k[3:]] = v
            elif k.startswith("c2_"):
                params2[k[3:]] = v
        return params1, params2
    
    def select_action(self, state: np.ndarray, exploration_noise: float = 0.1) -> np.ndarray:
        """选择动作（完全对齐PyTorch版本的噪声逻辑）"""
        state = state.reshape(1, -1).astype(DTYPE)
        action = self.actor.forward(state).flatten()
        
        if exploration_noise != 0:
            # 对齐PyTorch的torch.randn_like + clamp
            noise = np.random.normal(0, exploration_noise, size=self.action_dim).astype(DTYPE)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def _compute_critic_grads(self, state_batch: np.ndarray, action_batch: np.ndarray, target_Q: np.ndarray) -> Dict[str, np.ndarray]:
        """计算Critic梯度（严格复刻PyTorch自动微分）"""
        state_batch = state_batch.astype(DTYPE)
        action_batch = action_batch.astype(DTYPE)
        target_Q = target_Q.astype(DTYPE)
        
        # 前向传播并保存中间结果
        x = np.concatenate([state_batch, action_batch], axis=1)
        
        # 网络1前向
        h1_1 = relu(np.dot(x, self.critic.params1["w1"]) + self.critic.params1["b1"])
        h2_1 = relu(np.dot(h1_1, self.critic.params1["w2"]) + self.critic.params1["b2"])
        h3_1 = relu(np.dot(h2_1, self.critic.params1["w3"]) + self.critic.params1["b3"])
        q1 = np.dot(h3_1, self.critic.params1["w4"]) + self.critic.params1["b4"]
        
        # 网络2前向
        h1_2 = relu(np.dot(x, self.critic.params2["w1"]) + self.critic.params2["b1"])
        h2_2 = relu(np.dot(h1_2, self.critic.params2["w2"]) + self.critic.params2["b2"])
        h3_2 = relu(np.dot(h2_2, self.critic.params2["w3"]) + self.critic.params2["b3"])
        q2 = np.dot(h3_2, self.critic.params2["w4"]) + self.critic.params2["b4"]
        
        # MSE损失梯度（对齐PyTorch的nn.MSELoss）
        grad_q1 = 2 * (q1 - target_Q) / self.batch_size
        grad_q2 = 2 * (q2 - target_Q) / self.batch_size
        
        # -------------------------- 网络1梯度计算 --------------------------
        # w4/b4梯度
        grad_w4_1 = np.dot(h3_1.T, grad_q1)
        grad_b4_1 = np.sum(grad_q1, axis=0)
        
        # w3/b3梯度
        grad_h3_1 = np.dot(grad_q1, self.critic.params1["w4"].T)
        grad_h3_1 = grad_h3_1 * (h3_1 > 0).astype(DTYPE)  # ReLU梯度
        grad_w3_1 = np.dot(h2_1.T, grad_h3_1)
        grad_b3_1 = np.sum(grad_h3_1, axis=0)
        
        # w2/b2梯度
        grad_h2_1 = np.dot(grad_h3_1, self.critic.params1["w3"].T)
        grad_h2_1 = grad_h2_1 * (h2_1 > 0).astype(DTYPE)
        grad_w2_1 = np.dot(h1_1.T, grad_h2_1)
        grad_b2_1 = np.sum(grad_h2_1, axis=0)
        
        # w1/b1梯度
        grad_h1_1 = np.dot(grad_h2_1, self.critic.params1["w2"].T)
        grad_h1_1 = grad_h1_1 * (h1_1 > 0).astype(DTYPE)
        grad_w1_1 = np.dot(x.T, grad_h1_1)
        grad_b1_1 = np.sum(grad_h1_1, axis=0)
        
        # -------------------------- 网络2梯度计算 --------------------------
        # w4/b4梯度
        grad_w4_2 = np.dot(h3_2.T, grad_q2)
        grad_b4_2 = np.sum(grad_q2, axis=0)
        
        # w3/b3梯度
        grad_h3_2 = np.dot(grad_q2, self.critic.params2["w4"].T)
        grad_h3_2 = grad_h3_2 * (h3_2 > 0).astype(DTYPE)
        grad_w3_2 = np.dot(h2_2.T, grad_h3_2)
        grad_b3_2 = np.sum(grad_h3_2, axis=0)
        
        # w2/b2梯度
        grad_h2_2 = np.dot(grad_h3_2, self.critic.params2["w3"].T)
        grad_h2_2 = grad_h2_2 * (h2_2 > 0).astype(DTYPE)
        grad_w2_2 = np.dot(h1_2.T, grad_h2_2)
        grad_b2_2 = np.sum(grad_h2_2, axis=0)
        
        # w1/b1梯度
        grad_h1_2 = np.dot(grad_h2_2, self.critic.params2["w2"].T)
        grad_h1_2 = grad_h1_2 * (h1_2 > 0).astype(DTYPE)
        grad_w1_2 = np.dot(x.T, grad_h1_2)
        grad_b1_2 = np.sum(grad_h1_2, axis=0)
        
        # 整合梯度（扁平化，与参数键匹配）
        grads = {
            # 网络1梯度
            "c1_w1": grad_w1_1, "c1_b1": grad_b1_1,
            "c1_w2": grad_w2_1, "c1_b2": grad_b2_1,
            "c1_w3": grad_w3_1, "c1_b3": grad_b3_1,
            "c1_w4": grad_w4_1, "c1_b4": grad_b4_1,
            # 网络2梯度
            "c2_w1": grad_w1_2, "c2_b1": grad_b1_2,
            "c2_w2": grad_w2_2, "c2_b2": grad_b2_2,
            "c2_w3": grad_w3_2, "c2_b3": grad_b3_2,
            "c2_w4": grad_w4_2, "c2_b4": grad_b4_2,
        }
        
        return grads
    
    def _compute_actor_grads(self, state_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """计算Actor梯度（修复维度不匹配问题）"""
        state_batch = state_batch.astype(DTYPE)
        batch_size = state_batch.shape[0]
        
        # -------------------------- Actor前向传播（保存中间结果） --------------------------
        # 第一层
        h1 = np.dot(state_batch, self.actor.params["w1"]) + self.actor.params["b1"]
        h1_relu = relu(h1)
        # 第二层
        h2 = np.dot(h1_relu, self.actor.params["w2"]) + self.actor.params["b2"]
        h2_relu = relu(h2)
        # 第三层
        h3 = np.dot(h2_relu, self.actor.params["w3"]) + self.actor.params["b3"]
        h3_relu = relu(h3)
        # 输出层
        logits = np.dot(h3_relu, self.actor.params["w4"]) + self.actor.params["b4"]
        tanh_out = tanh(logits)
        action = self.max_action * tanh_out
        
        # -------------------------- Critic Q1前向传播 --------------------------
        # 计算Q值
        q1 = self.critic.Q1(state_batch, action)
        # Actor损失：-mean(Q1)
        loss = -np.mean(q1)
        # 损失对Q1的梯度
        grad_q1 = -1.0 / batch_size * np.ones_like(q1, dtype=DTYPE)
        
        # -------------------------- Critic反向传播到action --------------------------
        # 拼接state和action
        x = np.concatenate([state_batch, action], axis=1)
        # Critic网络1前向（保存中间结果）
        c_h1 = np.dot(x, self.critic.params1["w1"]) + self.critic.params1["b1"]
        c_h1_relu = relu(c_h1)
        c_h2 = np.dot(c_h1_relu, self.critic.params1["w2"]) + self.critic.params1["b2"]
        c_h2_relu = relu(c_h2)
        c_h3 = np.dot(c_h2_relu, self.critic.params1["w3"]) + self.critic.params1["b3"]
        c_h3_relu = relu(c_h3)
        
        # Critic反向传播
        # 对c_h3_relu的梯度
        grad_c_h3 = np.dot(grad_q1, self.critic.params1["w4"].T)
        grad_c_h3 = grad_c_h3 * (c_h3_relu > 0).astype(DTYPE)
        # 对c_h2_relu的梯度
        grad_c_h2 = np.dot(grad_c_h3, self.critic.params1["w3"].T)
        grad_c_h2 = grad_c_h2 * (c_h2_relu > 0).astype(DTYPE)
        # 对c_h1_relu的梯度
        grad_c_h1 = np.dot(grad_c_h2, self.critic.params1["w2"].T)
        grad_c_h1 = grad_c_h1 * (c_h1_relu > 0).astype(DTYPE)
        # 对x的梯度
        grad_x = np.dot(grad_c_h1, self.critic.params1["w1"].T)
        # 提取对action的梯度（x的后action_dim列）
        grad_action = grad_x[:, -self.action_dim:]
        
        # -------------------------- Actor反向传播 --------------------------
        # 对tanh_out的梯度
        grad_tanh = grad_action * self.max_action
        # tanh导数：1 - tanh^2
        tanh_deriv = (1 - np.square(tanh_out)).astype(DTYPE)
        grad_logits = grad_tanh * tanh_deriv
        
        # 对w4/b4的梯度
        grad_w4 = np.dot(h3_relu.T, grad_logits)
        grad_b4 = np.sum(grad_logits, axis=0)
        
        # 对h3_relu的梯度
        grad_h3 = np.dot(grad_logits, self.actor.params["w4"].T)
        grad_h3 = grad_h3 * (h3_relu > 0).astype(DTYPE)
        # 对w3/b3的梯度
        grad_w3 = np.dot(h2_relu.T, grad_h3)
        grad_b3 = np.sum(grad_h3, axis=0)
        
        # 对h2_relu的梯度
        grad_h2 = np.dot(grad_h3, self.actor.params["w3"].T)
        grad_h2 = grad_h2 * (h2_relu > 0).astype(DTYPE)
        # 对w2/b2的梯度
        grad_w2 = np.dot(h1_relu.T, grad_h2)
        grad_b2 = np.sum(grad_h2, axis=0)
        
        # 对h1_relu的梯度
        grad_h1 = np.dot(grad_h2, self.actor.params["w2"].T)
        grad_h1 = grad_h1 * (h1_relu > 0).astype(DTYPE)
        # 对w1/b1的梯度
        grad_w1 = np.dot(state_batch.T, grad_h1)
        grad_b1 = np.sum(grad_h1, axis=0)
        
        return {
            "w1": grad_w1, "b1": grad_b1,
            "w2": grad_w2, "b2": grad_b2,
            "w3": grad_w3, "b3": grad_b3,
            "w4": grad_w4, "b4": grad_b4
        }
    
    def train(self):
        """训练一步（完全对齐PyTorch版本的训练逻辑）"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.total_it += 1
        
        # 采样批次数据（对齐PyTorch的random.sample）
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = np.array([e[0] for e in batch], dtype=DTYPE)
        action_batch = np.array([e[1] for e in batch], dtype=DTYPE)
        reward_batch = np.array([e[2] for e in batch], dtype=DTYPE).reshape(-1, 1)  # 对齐unsqueeze(1)
        next_state_batch = np.array([e[3] for e in batch], dtype=DTYPE)
        done_batch = np.array([e[4] for e in batch], dtype=DTYPE).reshape(-1, 1)
        
        # 计算目标Q值（对齐PyTorch的with torch.no_grad()）
        noise = np.random.normal(0, self.policy_noise, size=action_batch.shape).astype(DTYPE)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        next_actions = self.actor_target.forward(next_state_batch) + noise
        next_actions = np.clip(next_actions, -self.max_action, self.max_action)
        
        target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_actions)
        target_Q = np.minimum(target_Q1, target_Q2)
        target_Q = reward_batch + (1 - done_batch) * 0.99 * target_Q
        
        # -------------------------- 更新Critic --------------------------
        # 计算梯度
        critic_grads = self._compute_critic_grads(state_batch, action_batch, target_Q)
        # 梯度裁剪（对齐PyTorch）
        critic_grads = clip_grad_norm(critic_grads)
        # 获取当前Critic参数
        critic_flat_params = self._flatten_critic_params()
        # 执行优化
        updated_critic_params = self.critic_optimizer.step(critic_flat_params, critic_grads)
        # 更新Critic参数
        params1, params2 = self._unflatten_critic_params(updated_critic_params)
        self.critic.set_params(params1, params2)
        
        # -------------------------- 延迟更新Actor --------------------------
        if self.total_it % self.policy_freq == 0:
            # 计算Actor梯度
            actor_grads = self._compute_actor_grads(state_batch)
            # 梯度裁剪
            actor_grads = clip_grad_norm(actor_grads)
            # 执行优化
            updated_actor_params = self.actor_optimizer.step(self.actor.get_params(), actor_grads)
            # 更新Actor参数
            self.actor.set_params(updated_actor_params)
            
            # 软更新目标网络（完全对齐PyTorch的copy_逻辑）
            # 更新Critic Target
            critic_params1, critic_params2 = self.critic.get_params()
            target_critic_params1, target_critic_params2 = self.critic_target.get_params()
            
            for k in critic_params1.keys():
                target_critic_params1[k] = 0.005 * critic_params1[k] + 0.995 * target_critic_params1[k]
            for k in critic_params2.keys():
                target_critic_params2[k] = 0.005 * critic_params2[k] + 0.995 * target_critic_params2[k]
            
            self.critic_target.set_params(target_critic_params1, target_critic_params2)
            
            # 更新Actor Target
            actor_params = self.actor.get_params()
            target_actor_params = self.actor_target.get_params()
            
            for k in actor_params.keys():
                target_actor_params[k] = 0.005 * actor_params[k] + 0.995 * target_actor_params[k]
            
            self.actor_target.set_params(target_actor_params)
    
    def save(self, filename: str):
        """保存模型（对齐PyTorch版本的保存逻辑）"""
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m%d_%H%M%S")
        save_filename = f"{filename}_{time_str}.npz"
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_filename)
        
        # 收集所有参数
        actor_params = self.actor.get_params()
        critic_params1, critic_params2 = self.critic.get_params()
        
        # 保存为npz文件（对齐PyTorch的pth保存）
        np.savez(
            save_path,
            **{f"actor_{k}": v for k, v in actor_params.items()},
            **{f"critic1_{k}": v for k, v in critic_params1.items()},
            **{f"critic2_{k}": v for k, v in critic_params2.items()}
        )
        
        print(f"最优模型已保存至: {save_path}")

# -------------------------- 路径配置（完全复用PyTorch版本） --------------------------
def get_project_paths() -> Dict[str, str]:
    """
    获取所有数据文件的路径（基于程序所在目录）
    Returns:
        包含各类文件路径的字典
    """
    # 获取程序所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义各目录路径
    power_data_dir = os.path.join(base_dir, "POWERdata", "C5336")
    hisdata_dir = os.path.join(power_data_dir, "train")
    modeldata_dir = os.path.join(power_data_dir, "modeldata")
    
    # 构建文件路径字典
    paths = {
        "base_dir": base_dir,
        "hisdata_dir": hisdata_dir,
        "volcst_file": os.path.join(modeldata_dir, "volcst_C5336.xlsx"),
        "kvnd_file": os.path.join(modeldata_dir, "kvnd_C5336.xlsx"),
        "branch_file": os.path.join(modeldata_dir, "branch_C5336.xlsx"),
        "pv_file": os.path.join(modeldata_dir, "pv_C5336.xlsx"),
        "model_save_dir": base_dir  # 模型保存到程序目录
    }
    
    # 检查目录是否存在
    for key, path in paths.items():
        if "dir" in key and not os.path.exists(path):
            raise FileNotFoundError(f"目录不存在: {path}")
    
    return paths

# -------------------------- 配置读取函数（完全复用PyTorch版本） --------------------------
def read_voltage_limits(volcst_file: str) -> Tuple[float, float]:
    """
    读取电压上下限（从volcst_C5336.xlsx）
    Args:
        volcst_file: 电压限值文件路径
    Returns:
        (电压下限, 电压上限)
    """
    df = pd.read_excel(volcst_file, header=None)
    # 第二行第一列=下限，第二行第二列=上限
    voltage_lower = float(df.iloc[1, 0])
    voltage_upper = float(df.iloc[1, 1])
    return voltage_lower, voltage_upper

def read_key_nodes(kvnd_file: str) -> List[int]:
    """
    读取关键节点索引（从kvnd_C5336.xlsx）
    Args:
        kvnd_file: 关键节点文件路径
    Returns:
        关键节点索引列表（节点号-1转换为索引）
    """
    df = pd.read_excel(kvnd_file, header=0)  # 第一行是列名
    # 从第二行第一列开始读取节点号，转换为索引（-1）
    key_node_nums = df.iloc[:, 0].dropna().astype(int).tolist()
    key_node_indices = [num - 1 for num in key_node_nums]  # 节点号转索引
    return key_node_indices

def read_branch_data(branch_file: str) -> np.ndarray:
    """
    读取支路数据（从branch_C5336.xlsx）
    【修改点1】返回二维数组（每行5列：线路号	首节点	末节点	电阻	电抗）
    Args:
        branch_file: 支路数据文件路径
    Returns:
        二维支路数据数组 (n_branches, 5)
    """
    df = pd.read_excel(branch_file, header=0)  # 列名：线路号	首节点	末节点	电阻	电抗
    # 按行读取，直接返回二维数组
    branch_data = []
    for _, row in df.iterrows():
        branch_data.append([
            row.iloc[0],  # 线路号
            row.iloc[1],  # 首节点
            row.iloc[2],  # 末节点
            row.iloc[3],  # 电阻
            row.iloc[4]   # 电抗
        ])
    # 【关键修改】返回二维数组，不再展平
    return np.array(branch_data)

def read_tunable_q_nodes(pv_file: str, bus_data: np.ndarray) -> List[Tuple[int, float, float, str]]:
    """
    读取并计算可调节无功节点配置（从pv_C5336.xlsx）
    Args:
        pv_file: 光伏节点文件路径
        bus_data: Bus矩阵数据（用于获取当前有功值）
    Returns:
        TUNABLE_Q_NODES格式的列表：[(节点索引, 无功下限, 无功上限, 节点名称), ...]
    """
    df = pd.read_excel(pv_file, header=0)  # 列名：节点号 容量 调度命名
    tunable_q_nodes = []
    
    for _, row in df.iterrows():
        # 读取基础信息
        node_num = int(row.iloc[0])  # 光伏节点号
        capacity = float(row.iloc[1])  # 容量
        node_name = row.iloc[2] if not pd.isna(row.iloc[2]) else f"节点{node_num}"
        
        # 计算可调无功上下限
        node_index = node_num - 1  # 转索引
        p_current = bus_data[node_index, 1]  # Bus矩阵中该节点的有功值
        q_max = np.sqrt(max(0, capacity**2 - p_current**2))  # 无功上限
        q_min = -q_max  # 无功下限
        
        # 添加到列表
        tunable_q_nodes.append((node_index, q_min, q_max, node_name))
    
    return tunable_q_nodes

def read_training_sample(hisdata_dir: str, sample_file: str = None) -> Tuple[np.ndarray, float]:
    """
    读取训练样本数据（从hisdata/rltrain下的Excel文件）
    Args:
        hisdata_dir: 训练样本目录
        sample_file: 指定样本文件（None则取第一个）
    Returns:
        (Bus矩阵数据, 基准电压UB)
    """
    # 获取所有样本文件
    sample_files = glob.glob(os.path.join(hisdata_dir, "C5336_*.xlsx"))
    if not sample_files:
        raise FileNotFoundError(f"未找到训练样本文件: {hisdata_dir}")
    
    # 选择样本文件（指定或第一个）
    if sample_file and os.path.exists(sample_file):
        file_path = sample_file
    else:
        file_path = sample_files[0]
    
    # 读取Bus数据（sheet=bus）
    df_bus = pd.read_excel(file_path, sheet_name="bus", header=0)  # 列名：节点号 有功值 无功值
    bus_data = []
    for _, row in df_bus.iterrows():
        bus_data.append([
            int(row.iloc[0]),    # 节点号
            float(row.iloc[1]),  # 有功值
            float(row.iloc[2])   # 无功值
        ])
    # 【确保Bus是二维数组】
    bus_array = np.array(bus_data)
    
    # 读取基准电压UB（sheet=slack）
    df_slack = pd.read_excel(file_path, sheet_name="slack")
    ub = float(df_slack.iloc[0, 0])  # slack sheet的第一个值
    
    return bus_array, ub

# -------------------------- 模型保存优化（完全复用PyTorch版本） --------------------------
def save_best_normalized_model(agent, filename: str, best_normalized_improvement: float, 
                               current_normalized_improvement: float, paths: Dict[str, str]):
    """
    基于归一化改善程度保存最优模型
    Args:
        current_normalized_improvement: 当前轮次的平均归一化改善程度
                                        （正数表示改善，越大越好）
    Returns:
        更新后的最优归一化改善程度
    """
    # 归一化改善程度越大越好（表示改善越多）
    if current_normalized_improvement > best_normalized_improvement:
        # 删除旧模型
        old_models = glob.glob(os.path.join(paths["model_save_dir"], f"{filename}_*.npz"))
        for old_model in old_models:
            try:
                os.remove(old_model)
            except Exception as e:
                print(f"删除旧模型失败: {old_model}, 错误: {e}")
        
        # 保存新模型
        agent.save(filename)
        best_normalized_improvement = current_normalized_improvement
        print(f"📈 更新最优模型，平均归一化改善: {best_normalized_improvement:.2f}%")
    
    return best_normalized_improvement

# -------------------------- 改进的强化学习环境（自动计算基准网损率） --------------------------
class ImprovedPowerSystemEnv:
    def __init__(self, 
                 bus_data: np.ndarray,
                 branch_data: np.ndarray,
                 voltage_lower: float,
                 voltage_upper: float,
                 key_nodes: List[int],
                 tunable_q_nodes: List[Tuple[int, float, float, str]],
                 ub: float,  # 【修改点2】新增UB参数，初始化时直接传入
                 baseline_info: Dict[str, float] = None,  # 新增：基准信息
                 sb: float = 10.0,
                 pr: float = 1e-6):
        """
        初始化电力系统环境（改进版本，支持基准网损率）
        Args:
            bus_data: Bus矩阵数据
            branch_data: 支路数据（二维数组）
            voltage_lower: 电压下限
            voltage_upper: 电压上限
            key_nodes: 关键节点索引列表
            tunable_q_nodes: 可调节无功节点配置
            ub: 基准电压（从训练样本读取）
            baseline_info: 基准信息字典，包含baseline_loss（基准网损率）和total_power（总功率）
            sb: 基准功率 MVA
            pr: 潮流收敛精度
        """
        self.Bus = bus_data
        self.Branch = branch_data  # 现在是二维数组 (n_branches, 5)
        self.SB = sb
        self.UB = ub  # 【关键修改】初始化时直接赋值，避免None
        self.pr = pr
        
        self.state_dim = len(key_nodes)
        self.action_dim = len(tunable_q_nodes)
        self.max_action = 1.0
        
        # 无功上下限
        self.q_mins = np.array([node[1] for node in tunable_q_nodes])
        self.q_maxs = np.array([node[2] for node in tunable_q_nodes])
        self.tunable_q_nodes = tunable_q_nodes
        
        self.v_min = voltage_lower
        self.v_max = voltage_upper
        self.key_nodes = key_nodes
        
        self.previous_loss_rate = None
        self.previous_voltages = None
        self.previous_actions = None
        
        # 基准信息（如果没有提供，自动计算）
        if baseline_info is None:
            self.baseline_info = self._calculate_baseline_info()
        else:
            self.baseline_info = baseline_info
        
        # 计算归一化参数
        self._calculate_normalization_params()
        
        self.reset()  # 现在UB已初始化，reset不会报错
    
    def _calculate_baseline_info(self) -> Dict[str, float]:
        """自动计算基准网损率和总功率"""
        # 使用初始无功值计算潮流（即不调节）
        initial_q = [self.Bus[node[0], 2] for node in self.tunable_q_nodes]
        loss_rate, voltages, power_info = self._power_flow(initial_q)
        
        if loss_rate is None or power_info is None:
            # 如果初始潮流不收敛，估算基准值
            total_input_power = self._estimate_total_power()
            baseline_loss = 10.0  # 默认基准网损率
        else:
            total_input_power = power_info[2]  # 总输入功率
            baseline_loss = loss_rate
        
        return {
            'baseline_loss': baseline_loss,
            'total_power': total_input_power
        }
    
    def _estimate_total_power(self) -> float:
        """估算总输入功率（如果潮流计算失败时使用）"""
        # 简单估算：所有负荷节点的有功绝对值之和 * 1.1
        load_power = np.sum(np.abs(self.Bus[self.Bus[:, 1] > 0, 1]))  # 负荷节点有功
        return load_power * 1.1  # 乘以系数考虑网损
    
    def _calculate_normalization_params(self):
        """基于基准信息计算归一化参数"""
        baseline_loss = self.baseline_info['baseline_loss']
        total_power = self.baseline_info['total_power']
        
        # 1. 基于功率的权重（自动计算）
        # 高功率样本权重大，低功率样本权重小
        self.power_weight = np.clip(total_power / 100.0, 0.5, 2.0)
        
        # 2. 基于基准网损率的难度系数
        # 基准网损率高的样本，优化空间大，难度低
        # 基准网损率低的样本，优化空间小，难度高
        if baseline_loss > 8.0:
            self.difficulty_factor = 0.8  # 容易
        elif baseline_loss > 5.0:
            self.difficulty_factor = 1.0  # 中等
        elif baseline_loss > 3.0:
            self.difficulty_factor = 1.2  # 较难
        else:
            self.difficulty_factor = 1.5  # 很难
    
    def reset(self):
        """重置环境，计算初始状态"""
        # 从初始潮流计算获取初始电压
        initial_q = [self.Bus[node[0], 2] for node in self.tunable_q_nodes]
        _, initial_voltages, power_info = self._power_flow(initial_q)
        
        if initial_voltages is None:
            initial_voltages = np.ones(self.Bus.shape[0]) * self.UB
        
        self.state = self._build_state(initial_voltages)
        self.previous_loss_rate = None
        self.previous_voltages = None
        self.previous_actions = None
        
        return self.state
    
    def _build_state(self, voltages):
        """构建归一化的状态向量"""
        key_voltages = voltages[self.key_nodes]
        normalized_voltages = (key_voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        return normalized_voltages
    
    def step(self, action):
        """执行动作，返回新状态和奖励"""
        actual_action = self._denormalize_action(action)
        loss_rate, voltages, power_info = self._power_flow(actual_action)
        
        if loss_rate is None:
            reward = -200  # 潮流不收敛惩罚加重
            done = True
            next_state = self.state
            info = {
                "loss_rate": 100, 
                "voltages": np.ones(self.Bus.shape[0]) * self.UB,
                "normalized_improvement": -100,
                "relative_improvement": -100
            }
        else:
            # 计算归一化奖励
            reward, normalized_improvement, relative_improvement = self._calculate_normalized_reward(
                loss_rate, voltages, actual_action
            )
            
            done = False
            next_state = self._build_state(voltages)
            self.state = next_state
            self.previous_loss_rate = loss_rate
            self.previous_voltages = voltages.copy()
            self.previous_actions = actual_action
            
            info = {
                "loss_rate": loss_rate,
                "voltages": voltages,
                "normalized_improvement": normalized_improvement,
                "relative_improvement": relative_improvement,
                "baseline_loss": self.baseline_info['baseline_loss']
            }
        
        return next_state, reward, done, info
    
    def _denormalize_action(self, action):
        """反归一化动作到实际无功值范围"""
        actual_actions = []
        for i in range(len(action)):
            normalized = np.clip(action[i], -1, 1)
            q_min = self.q_mins[i]
            q_max = self.q_maxs[i]
            actual = (normalized + 1) / 2 * (q_max - q_min) + q_min
            actual_actions.append(actual)
        return actual_actions
    
    def _calculate_normalized_reward(self, loss_rate: float, voltages: np.ndarray, 
                                    actual_action: List[float]) -> Tuple[float, float, float]:
        """
        核心：自动归一化的奖励计算
        返回：(总奖励, 归一化改善程度, 相对改善百分比)
        """
        baseline_loss = self.baseline_info['baseline_loss']
        
        # 1. 计算相对改善百分比（核心归一化指标）
        if baseline_loss > 0:
            relative_improvement = (baseline_loss - loss_rate) / baseline_loss * 100
        else:
            relative_improvement = 0
        
        # 2. 计算归一化改善程度（应用权重和难度系数）
        normalized_improvement = relative_improvement * self.power_weight * self.difficulty_factor
        
        # 3. 计算基础奖励（基于归一化改善程度）
        base_reward = normalized_improvement * 2.0  # 缩放系数
        
        # 4. 电压约束惩罚（硬约束）
        voltage_penalty = 0
        voltage_reward = 0
        for v in voltages:
            if v < self.v_min:
                voltage_penalty += 100 * (self.v_min - v)  # 电压越限严重惩罚
            elif v > self.v_max:
                voltage_penalty += 100 * (v - self.v_max)
            else:
                # 电压在正常范围内，给予奖励，鼓励接近最优
                optimal_voltage = (self.v_min + self.v_max) / 2
                distance = abs(v - optimal_voltage) / optimal_voltage
                voltage_reward += 5 * (1 - distance)  # 越接近最优，奖励越高
        
        # 5. 动作平滑性奖励（鼓励小幅调节）
        action_penalty = 0
        if self.previous_actions is not None:
            # 计算动作变化量
            action_change = np.sum(np.abs(np.array(actual_action) - np.array(self.previous_actions)))
            # 鼓励平滑调节，大幅变化有惩罚
            if action_change > self.max_action * 0.5:  # 变化超过50%
                action_penalty = 5 * (action_change - self.max_action * 0.5)
        
        # 6. 收敛性奖励（鼓励快速改善）
        convergence_bonus = 0
        if self.previous_loss_rate is not None:
            # 计算改善速度
            improvement_speed = (self.previous_loss_rate - loss_rate) / max(self.previous_loss_rate, 1.0)
            if improvement_speed > 0.1:  # 改善超过10%
                convergence_bonus = 10 * improvement_speed
        
        # 7. 综合奖励
        total_reward = (
            base_reward + 
            voltage_reward + 
            convergence_bonus - 
            voltage_penalty - 
            action_penalty
        )
        
        # 8. 特别奖励：如果满足所有约束且有显著改善
        if (voltage_penalty == 0 and 
            relative_improvement > 10 and  # 相对改善超过10%
            action_penalty < 2):  # 动作平滑
            total_reward += 20
        
        return total_reward, normalized_improvement, relative_improvement
    
    def _power_flow(self, tunable_q_values):
        """内部潮流计算函数（适配通用化参数）"""
        Bus_copy = copy.deepcopy(self.Bus)
        Branch_copy = copy.deepcopy(self.Branch)  # 现在是二维数组
        
        # 更新可调无功节点值
        for i, (node_idx, _, _, _) in enumerate(self.tunable_q_nodes):
            Bus_copy[node_idx, 2] = tunable_q_values[i]
        
        # 标幺值转换（现在Branch_copy是二维数组，索引正常）
        Bus_copy[:, 1] = Bus_copy[:, 1] / self.SB
        Bus_copy[:, 2] = Bus_copy[:, 2] / self.SB
        Branch_copy[:, 3] = Branch_copy[:, 3] * self.SB / (self.UB **2)  # 电阻标幺值
        Branch_copy[:, 4] = Branch_copy[:, 4] * self.SB / (self.UB** 2)  # 电抗标幺值
        
        busnum = Bus_copy.shape[0]
        branchnum = Branch_copy.shape[0]
        
        # 节点类型判断
        node_types = []
        for i in range(busnum):
            node_id = Bus_copy[i, 0]
            p = Bus_copy[i, 1]
            if node_id == 1:
                node_types.append("平衡节点")
            elif p < 0:
                node_types.append("光伏节点")
            elif p > 0:
                node_types.append("负荷节点")
            else:
                node_types.append("普通节点")
        
        # 初始化电压和相角
        Vbus = np.ones(busnum)
        Vbus[0] = 1.0
        cita = np.zeros(busnum)
        
        k = 0
        Ploss = np.zeros(branchnum)
        Qloss = np.zeros(branchnum)
        P = np.zeros(branchnum)
        Q = np.zeros(branchnum)
        F = 0
        
        # 支路排序（适配二维Branch数据）
        TempBranch = Branch_copy.copy()
        s1 = np.zeros((0, 5))
        while TempBranch.size > 0:
            s = TempBranch.shape[0] - 1
            s2 = np.zeros((0, 5))
            while s >= 0:
                # 查找以当前支路末节点为首节点的支路
                i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
                if i.size == 0:
                    # 没有后续支路，加入s1
                    if s1.size == 0:
                        s1 = TempBranch[s, :].reshape(1, -1)
                    else:
                        s1 = np.vstack([s1, TempBranch[s, :]])
                else:
                    # 有后续支路，加入s2
                    if s2.size == 0:
                        s2 = TempBranch[s, :].reshape(1, -1)
                    else:
                        s2 = np.vstack([s2, TempBranch[s, :]])
                s -= 1
            TempBranch = s2.copy()
        
        # 潮流迭代计算
        while k < 100 and F == 0:
            Pij1 = np.zeros(busnum)
            Qij1 = np.zeros(busnum)
            
            for s in range(branchnum):
                ii = int(s1[s, 1] - 1)  # 首节点索引
                jj = int(s1[s, 2] - 1)  # 末节点索引
                Pload = Bus_copy[jj, 1]
                Qload = Bus_copy[jj, 2]
                R = s1[s, 3]
                X = s1[s, 4]
                VV = Vbus[jj]
                
                Pij0 = Pij1[jj]
                Qij0 = Qij1[jj]
                
                II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
                Ploss[s] = II * R
                Qloss[s] = II * X
                
                P[s] = Pload + Ploss[s] + Pij0
                Q[s] = Qload + Qloss[s] + Qij0
                
                Pij1[ii] += P[s]
                Qij1[ii] += Q[s]
            
            # 电压计算
            for s in range(branchnum-1, -1, -1):
                ii = int(s1[s, 2] - 1)  # 末节点索引
                kk = int(s1[s, 1] - 1)  # 首节点索引
                R = s1[s, 3]
                X = s1[s, 4]
                
                V_real = Vbus[kk] - (P[s]*R + Q[s]*X) / Vbus[kk]
                V_imag = (P[s]*X - Q[s]*R) / Vbus[kk]
                
                Vbus[ii] = np.sqrt(V_real**2 + V_imag**2)
                cita[ii] = cita[kk] - np.arctan2(V_imag, V_real)
            
            # 校验收敛
            Pij2 = np.zeros(busnum)
            Qij2 = np.zeros(busnum)
            for s in range(branchnum):
                ii = int(s1[s, 1] - 1)
                jj = int(s1[s, 2] - 1)
                Pload = Bus_copy[jj, 1]
                Qload = Bus_copy[jj, 2]
                R = s1[s, 3]
                X = s1[s, 4]
                VV = Vbus[jj]
                
                Pij0 = Pij2[jj]
                Qij0 = Qij2[jj]
                
                II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
                P_val = Pload + II * R + Pij0
                Q_val = Qload + II * X + Qij0
                
                Pij2[ii] += P_val
                Qij2[ii] += Q_val
            
            ddp = np.max(np.abs(Pij1 - Pij2))
            ddq = np.max(np.abs(Qij1 - Qij2))
            if ddp < self.pr and ddq < self.pr:
                F = 1
            k += 1
        
        # 收敛判断
        if k == 100:
            return None, None, None
        
        # 计算网损率
        P1 = np.sum(Ploss)
        balance_node_output = Pij2[0] * self.SB
        pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
        pv_total_injection = sum(-Bus_copy[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * self.SB
        total_input_power = balance_node_output + pv_total_injection
        
        load_nodes_mask = [typ == "负荷节点" for typ in node_types]
        total_output_power = sum(Bus_copy[i, 1] for i in range(busnum) if load_nodes_mask[i]) * self.SB
        
        loss_rate = (total_input_power - total_output_power) / total_input_power * 100 if total_input_power != 0 else 0.0
        Vbus_kv = Vbus * self.UB
        
        return loss_rate, Vbus_kv, (balance_node_output, pv_total_injection, total_input_power, total_output_power)

# -------------------------- 改进的样本管理器（自动计算基准网损率） --------------------------
class ImprovedSampleManager:
    """改进的样本管理器，自动计算所有样本的基准网损率"""
    def __init__(self, paths: Dict[str, str]):
        """
        初始化样本管理器
        Args:
            paths: 路径字典
        """
        self.paths = paths
        self.hisdata_dir = paths["hisdata_dir"]
        
        # 获取所有样本文件
        self.sample_files = glob.glob(os.path.join(self.hisdata_dir, "C5336_*.xlsx"))
        if not self.sample_files:
            raise FileNotFoundError(f"未找到训练样本文件: {self.hisdata_dir}")
        
        print(f"找到 {len(self.sample_files)} 个训练样本")
        
        # 自动计算所有样本的基准网损率
        self.sample_baselines = self._calculate_all_baselines()
        
        # 计算归一化统计量
        self._calculate_normalization_statistics()
    
    def get_sample_data(self, sample_file: str) -> Tuple[np.ndarray, float, List[Tuple[int, float, float, str]]]:
        """
        获取指定样本的数据
        Args:
            sample_file: 样本文件路径
        Returns:
            (bus_data, ub, tunable_q_nodes)
        """
        # 读取Bus数据和UB
        bus_data, ub = read_training_sample(self.paths["hisdata_dir"], sample_file)
        
        # 读取并计算可调无功节点
        tunable_q_nodes = read_tunable_q_nodes(self.paths["pv_file"], bus_data)
        
        return bus_data, ub, tunable_q_nodes
    
    def _calculate_baseline_for_sample(self, sample_file: str) -> Dict[str, float]:
        """
        计算单个样本的基准信息
        Returns: 包含baseline_loss和total_power的字典
        """
        # 读取样本数据
        bus_data, ub = read_training_sample(self.paths["hisdata_dir"], sample_file)
        tunable_q_nodes = read_tunable_q_nodes(self.paths["pv_file"], bus_data)
        
        # 读取基础配置（临时）
        voltage_lower, voltage_upper = read_voltage_limits(self.paths["volcst_file"])
        key_nodes = read_key_nodes(self.paths["kvnd_file"])
        branch_data = read_branch_data(self.paths["branch_file"])
        
        # 创建临时环境计算基准
        temp_env = ImprovedPowerSystemEnv(
            bus_data=bus_data,
            branch_data=branch_data,
            voltage_lower=voltage_lower,
            voltage_upper=voltage_upper,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_q_nodes,
            ub=ub
        )
        
        return temp_env.baseline_info
    
    def _calculate_all_baselines(self) -> Dict[str, Dict[str, float]]:
        """计算所有样本的基准信息"""
        baselines = {}
        
        print("自动计算所有样本的基准网损率...")
        for i, sample_file in enumerate(self.sample_files):
            sample_name = os.path.basename(sample_file)
            try:
                baseline_info = self._calculate_baseline_for_sample(sample_file)
                baselines[sample_file] = {
                    'name': sample_name,
                    'baseline_loss': baseline_info['baseline_loss'],
                    'total_power': baseline_info['total_power']
                }
                print(f"  样本 {i+1}/{len(self.sample_files)} ({sample_name}): "
                      f"基准网损率={baseline_info['baseline_loss']:.4f}%, "
                      f"总功率={baseline_info['total_power']:.2f}MW")
            except Exception as e:
                print(f"  样本 {sample_name} 基准计算失败: {e}")
                # 使用默认值
                baselines[sample_file] = {
                    'name': sample_name,
                    'baseline_loss': 10.0,
                    'total_power': 100.0
                }
        
        return baselines
    
    def _calculate_normalization_statistics(self):
        """计算基准统计信息"""
        if not self.sample_baselines:
            return
        
        baseline_values = [v['baseline_loss'] for v in self.sample_baselines.values()]
        power_values = [v['total_power'] for v in self.sample_baselines.values()]
        
        print(f"\n基准网损率统计:")
        print(f"  最小值: {min(baseline_values):.4f}%")
        print(f"  最大值: {max(baseline_values):.4f}%")
        print(f"  平均值: {np.mean(baseline_values):.4f}%")
        print(f"  中位数: {np.median(baseline_values):.4f}%")
        print(f"  标准差: {np.std(baseline_values):.4f}%")
        
        print(f"\n总输入功率统计:")
        print(f"  最小值: {min(power_values):.2f}MW")
        print(f"  最大值: {max(power_values):.2f}MW")
        print(f"  平均值: {np.mean(power_values):.2f}MW")
        print(f"  中位数: {np.median(power_values):.2f}MW")
    
    def create_env_for_sample(self, 
                             sample_file: str,
                             branch_data: np.ndarray,
                             voltage_lower: float,
                             voltage_upper: float,
                             key_nodes: List[int]) -> ImprovedPowerSystemEnv:
        """
        为指定样本创建环境，包含自动计算的基准信息
        """
        bus_data, ub, tunable_q_nodes = self.get_sample_data(sample_file)
        
        # 获取该样本的基准信息
        if sample_file in self.sample_baselines:
            baseline_info = {
                'baseline_loss': self.sample_baselines[sample_file]['baseline_loss'],
                'total_power': self.sample_baselines[sample_file]['total_power']
            }
        else:
            # 如果没有预先计算，现场计算
            baseline_info = self._calculate_baseline_for_sample(sample_file)
        
        env = ImprovedPowerSystemEnv(
            bus_data=bus_data,
            branch_data=branch_data,
            voltage_lower=voltage_lower,
            voltage_upper=voltage_upper,
            key_nodes=key_nodes,
            tunable_q_nodes=tunable_q_nodes,
            ub=ub,
            baseline_info=baseline_info  # 传入基准信息
        )
        
        return env
    
    def get_all_sample_files(self) -> List[str]:
        """获取所有样本文件路径"""
        return self.sample_files
    
    def get_baseline_stats(self) -> Dict[str, float]:
        """获取基准统计信息"""
        if not self.sample_baselines:
            return {}
        
        baseline_values = [v['baseline_loss'] for v in self.sample_baselines.values()]
        
        return {
            'min': min(baseline_values),
            'max': max(baseline_values),
            'mean': np.mean(baseline_values),
            'median': np.median(baseline_values),
            'std': np.std(baseline_values)
        }

# -------------------------- 改进的分层训练函数（基于归一化改善） --------------------------
def train_rl_model_with_normalized_improvement():
    """基于归一化改善的多样本分层训练"""
    # -------------------------- 1. 读取所有配置和数据 --------------------------
    print("=== 读取配置文件 ===")
    paths = get_project_paths()
    
    # 读取基础配置
    voltage_lower, voltage_upper = read_voltage_limits(paths["volcst_file"])
    key_nodes = read_key_nodes(paths["kvnd_file"])
    branch_data = read_branch_data(paths["branch_file"])  # 现在是二维数组
    
    # 初始化改进的样本管理器（自动计算基准网损率）
    sample_manager = ImprovedSampleManager(paths)
    sample_files = sample_manager.get_all_sample_files()
    
    # 获取基准统计信息
    baseline_stats = sample_manager.get_baseline_stats()
    
    # 打印配置信息
    print(f"电压上下限: [{voltage_lower:.2f}kV, {voltage_upper:.2f}kV]")
    print(f"关键节点索引: {key_nodes} (共{len(key_nodes)}个)")
    print(f"基准电压UB: 根据样本变化")
    print(f"支路数: {branch_data.shape[0]}")
    print(f"训练样本数: {len(sample_files)}")
    print(f"基准网损率中位数: {baseline_stats.get('median', 0):.4f}%")
    
    # -------------------------- 2. 初始化智能体 --------------------------
    print("\n=== 初始化智能体 ===")
    
    # 先用第一个样本获取状态和动作维度
    first_bus_data, first_ub, first_tunable_q_nodes = sample_manager.get_sample_data(sample_files[0])
    
    state_dim = len(key_nodes)
    action_dim = len(first_tunable_q_nodes)
    max_action = 1.0
    
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )
    
    # -------------------------- 3. 训练参数配置 --------------------------
    # 分层训练参数
    samples_per_epoch = 10  # 每个样本连续训练的episode数
    epochs = 30  # 训练轮数（每个样本训练samples_per_epoch个episode）
    
    max_steps = 3
    rewards_history = []
    normalized_improvements_history = []
    loss_rates_history = []
    best_normalized_improvement = -float('inf')  # 归一化改善程度越大越好
    
    print("\n=== 开始基于归一化改善的分层训练 ===")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"训练轮数: {epochs}, 每样本连续训练: {samples_per_epoch}个episode")
    print(f"每episode最大步数: {max_steps}")
    print(f"训练策略: 每个样本连续训练{samples_per_epoch}个episode后切换到下一个样本")
    print(f"奖励机制: 基于相对改善百分比，自动归一化")
    
    # -------------------------- 4. 预填充经验池 --------------------------
    print("预填充经验池...")
    prefill_steps = 0
    for sample_file in sample_files:
        if len(agent.replay_buffer) >= 5000:
            break
            
        env = sample_manager.create_env_for_sample(
            sample_file, branch_data, voltage_lower, voltage_upper, key_nodes
        )
        
        state = env.reset()
        for step in range(max_steps):
            action = np.random.uniform(-1, 1, size=action_dim)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            prefill_steps += 1
            
            if done or len(agent.replay_buffer) >= 5000:
                break
            state = next_state
    
    print(f"预填充完成，经验池大小: {len(agent.replay_buffer)}")
    
    # -------------------------- 5. 分层训练循环 --------------------------
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_normalized_improvements = []
        epoch_loss_rates = []
        
        # 遍历所有样本
        for sample_idx, sample_file in enumerate(sample_files):
            sample_name = os.path.basename(sample_file)
            
            # 为当前样本创建环境
            env = sample_manager.create_env_for_sample(
                sample_file, branch_data, voltage_lower, voltage_upper, key_nodes
            )
            
            # 在当前样本上连续训练samples_per_epoch个episode
            for episode_in_sample in range(samples_per_epoch):
                state = env.reset()
                episode_reward = 0
                episode_normalized_improvement = 0
                episode_loss_rate = 0
                
                for step in range(max_steps):
                    # 递减探索噪声（基于总训练进度）
                    total_episodes = epoch * len(sample_files) * samples_per_epoch + \
                                   sample_idx * samples_per_epoch + episode_in_sample
                    max_total_episodes = epochs * len(sample_files) * samples_per_epoch
                    exploration_noise = max(0.05, 0.3 * (1 - total_episodes / max_total_episodes))
                    
                    action = agent.select_action(state, exploration_noise)
                    next_state, reward, done, info = env.step(action)
                    
                    # 存储经验
                    agent.replay_buffer.append((state, action, reward, next_state, done))
                    
                    # 训练智能体
                    if len(agent.replay_buffer) > agent.batch_size:
                        agent.train()
                    
                    # 更新状态和奖励
                    state = next_state
                    episode_reward += reward
                    episode_normalized_improvement = info.get('normalized_improvement', 0)
                    episode_loss_rate = info.get('loss_rate', 100)
                    
                    if done:
                        break
                
                # 记录当前episode的结果
                epoch_rewards.append(episode_reward)
                epoch_normalized_improvements.append(episode_normalized_improvement)
                epoch_loss_rates.append(episode_loss_rate)
                
                # 打印进度
                if (episode_in_sample + 1) % 5 == 0 or episode_in_sample == samples_per_epoch - 1:
                    baseline_loss = info.get('baseline_loss', 0)
                    relative_improvement = info.get('relative_improvement', 0)
                    print(f"Epoch {epoch+1}/{epochs}, Sample {sample_idx+1}/{len(sample_files)} ({sample_name}), "
                          f"Episode {episode_in_sample+1}/{samples_per_epoch}: "
                          f"Reward={episode_reward:.2f}, Loss={episode_loss_rate:.2f}%, "
                          f"基准={baseline_loss:.2f}%, 相对改善={relative_improvement:.1f}%")
        
        # 计算本轮epoch的平均性能
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_normalized_improvement = np.mean(epoch_normalized_improvements) if epoch_normalized_improvements else 0
        avg_loss_rate = np.mean(epoch_loss_rates) if epoch_loss_rates else 100
        
        rewards_history.append(avg_reward)
        normalized_improvements_history.append(avg_normalized_improvement)
        loss_rates_history.append(avg_loss_rate)
        
        # 保存最优模型（基于归一化改善程度）
        best_normalized_improvement = save_best_normalized_model(
            agent=agent,
            filename="M1",
            best_normalized_improvement=best_normalized_improvement,
            current_normalized_improvement=avg_normalized_improvement,
            paths=paths
        )
        
        # 打印epoch总结
        print(f"\n=== Epoch {epoch+1}/{epochs} 完成 ===")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均归一化改善: {avg_normalized_improvement:.2f}%")
        print(f"平均网损率: {avg_loss_rate:.4f}%")
        print(f"当前最优归一化改善: {best_normalized_improvement:.2f}%")
        print("-" * 60)
    
    # -------------------------- 6. 绘制训练曲线 --------------------------
    plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 奖励曲线
    axes[0].plot(rewards_history, alpha=0.7, marker='o', markersize=3)
    axes[0].set_xlabel('训练轮次 (Epoch)')
    axes[0].set_ylabel('平均奖励')
    axes[0].set_title('多样本训练奖励曲线')
    axes[0].grid(True)
    
    # 归一化改善曲线
    axes[1].plot(normalized_improvements_history, alpha=0.7, marker='o', markersize=3, color='green')
    axes[1].set_xlabel('训练轮次 (Epoch)')
    axes[1].set_ylabel('平均归一化改善 (%)')
    axes[1].set_title('归一化改善程度变化')
    axes[1].grid(True)
    
    # 网损率曲线
    axes[2].plot(loss_rates_history, alpha=0.7, marker='o', markersize=3, color='red')
    axes[2].set_xlabel('训练轮次 (Epoch)')
    axes[2].set_ylabel('平均网损率 (%)')
    axes[2].set_title('网损率变化')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve_normalized_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # -------------------------- 7. 在所有样本上评估最终模型 --------------------------
    print("\n=== 在所有样本上评估最终模型 ===")
    
    evaluation_results = []
    for sample_idx, sample_file in enumerate(sample_files):
        sample_name = os.path.basename(sample_file)
        
        env = sample_manager.create_env_for_sample(
            sample_file, branch_data, voltage_lower, voltage_upper, key_nodes
        )
        
        state = env.reset()
        episode_reward = 0
        episode_normalized_improvement = 0
        episode_loss_rate = 0
        baseline_loss = env.baseline_info['baseline_loss']
        
        for step in range(max_steps):
            # 测试时不加探索噪声
            action = agent.select_action(state, exploration_noise=0)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_normalized_improvement = info.get('normalized_improvement', 0)
            episode_loss_rate = info.get('loss_rate', 100)
            
            if done:
                break
        
        relative_improvement = ((baseline_loss - episode_loss_rate) / baseline_loss * 100) if baseline_loss > 0 else 0
        
        evaluation_results.append({
            'sample': sample_name,
            'reward': episode_reward,
            'loss_rate': episode_loss_rate,
            'baseline_loss': baseline_loss,
            'relative_improvement': relative_improvement,
            'normalized_improvement': episode_normalized_improvement
        })
        
        print(f"样本 {sample_idx+1}/{len(sample_files)} ({sample_name}):")
        print(f"  基准网损率: {baseline_loss:.4f}%")
        print(f"  优化后网损率: {episode_loss_rate:.4f}%")
        print(f"  相对改善: {relative_improvement:.2f}%")
        print(f"  归一化改善: {episode_normalized_improvement:.2f}%")
        print(f"  奖励: {episode_reward:.2f}")
        print()
    
    # 计算平均性能
    avg_test_reward = np.mean([r['reward'] for r in evaluation_results])
    avg_test_loss_rate = np.mean([r['loss_rate'] for r in evaluation_results])
    avg_relative_improvement = np.mean([r['relative_improvement'] for r in evaluation_results])
    avg_normalized_improvement = np.mean([r['normalized_improvement'] for r in evaluation_results])
    
    print(f"\n=== 最终评估结果 ===")
    print(f"平均奖励: {avg_test_reward:.2f}")
    print(f"平均网损率: {avg_test_loss_rate:.4f}%")
    print(f"平均相对改善: {avg_relative_improvement:.2f}%")
    print(f"平均归一化改善: {avg_normalized_improvement:.2f}%")
    print(f"最优归一化改善: {best_normalized_improvement:.2f}%")
    print(f"仅保留最优模型文件，非最优模型已自动删除")
    
    # 保存评估结果到CSV文件
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv('evaluation_results_normalized.csv', index=False, encoding='utf-8-sig')
    print(f"评估结果已保存至: evaluation_results_normalized.csv")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 使用基于归一化改善的训练函数
        train_rl_model_with_normalized_improvement()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        raise