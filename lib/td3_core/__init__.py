#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3核心模块 - 更新后的导出
支持PyTorch版本和NumPy版本
"""
# PyTorch版本（原有）
from .models import ActorNetwork, CriticNetwork

# NumPy版本（新增）
from .numpy_models import ActorNetworkNumpy, CriticNetworkNumpy
from .numpy_backend import (
    AdamOptimizer, 
    relu, tanh, 
    xavier_uniform, 
    clip_grad_norm,
    soft_update,
    DTYPE
)

# 通用工具
from .power_flow import power_flow_calculation

__all__ = [
    # PyTorch版本
    'ActorNetwork', 
    'CriticNetwork',
    
    # NumPy版本
    'ActorNetworkNumpy',
    'CriticNetworkNumpy',
    'AdamOptimizer',
    
    # 激活函数
    'relu',
    'tanh',
    
    # 工具函数
    'xavier_uniform',
    'clip_grad_norm',
    'soft_update',
    
    # 通用
    'power_flow_calculation',
    'DTYPE'
]
