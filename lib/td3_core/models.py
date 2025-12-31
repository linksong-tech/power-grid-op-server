#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3神经网络模型定义
"""
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        return self.max_action * self.network(state)


class CriticNetwork(nn.Module):
    """Critic网络（价值网络，双Q网络）"""
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        
        self.network1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.network2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.network1(x), self.network2(x)
    
    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        return self.network1(x)
