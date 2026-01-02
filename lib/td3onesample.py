#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用化的电力系统无功优化TD3强化学习训练程序
支持从指定Excel文件读取配置参数和训练数据
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import datetime
import os
import glob
import pandas as pd
from typing import List, Tuple, Dict

# -------------------------- 路径配置（通用化核心） --------------------------
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

# -------------------------- 配置读取函数（通用化核心） --------------------------
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
    读取训练样本数据（从train下的Excel文件）
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

# -------------------------- 模型保存优化（只保留最优） --------------------------
def save_best_model(agent, filename: str, best_loss_rate: float, current_loss_rate: float, paths: Dict[str, str]):
    """
    只保存最优模型，删除旧的非最优模型
    Args:
        agent: TD3智能体
        filename: 模型基础名称
        best_loss_rate: 当前最优网损率
        current_loss_rate: 当前轮次网损率
        paths: 路径字典
    Returns:
        更新后的最优网损率
    """
    if current_loss_rate < best_loss_rate and current_loss_rate < 15:
        # 删除旧的最优模型
        old_models = glob.glob(os.path.join(paths["model_save_dir"], f"{filename}_*.pth"))
        for old_model in old_models:
            try:
                os.remove(old_model)
            except Exception as e:
                print(f"删除旧模型失败: {old_model}, 错误: {e}")
        
        # 保存新的最优模型
        agent.save(filename)
        best_loss_rate = current_loss_rate
        print(f"更新最优模型，网损率: {best_loss_rate:.4f}%")
    
    return best_loss_rate

# -------------------------- 网络定义（保持不变） --------------------------
class ActorNetwork(nn.Module):
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

# -------------------------- TD3算法（保持不变，优化保存提示） --------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 128
        
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
        
    def select_action(self, state, exploration_noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if exploration_noise != 0:
            noise = np.random.normal(0, exploration_noise, size=self.action_dim)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.total_it += 1
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        action_batch = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * 0.99 * target_Q
        
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def save(self, filename):
        # 1. 获取当前系统时间，并格式化为「月日_时分秒」
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m%d_%H%M%S")
        
        # 2. 拼接最终文件名
        save_filename = f"{filename}_{time_str}.pth"
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_filename)
        
        # 3. 保存模型
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, save_path)
        
        print(f"最优模型已保存至: {save_path}")

# -------------------------- 强化学习环境（适配通用化参数） --------------------------
class ImprovedPowerSystemEnv:
    def __init__(self, 
                 bus_data: np.ndarray,
                 branch_data: np.ndarray,
                 voltage_lower: float,
                 voltage_upper: float,
                 key_nodes: List[int],
                 tunable_q_nodes: List[Tuple[int, float, float, str]],
                 ub: float,  # 【修改点2】新增UB参数，初始化时直接传入
                 sb: float = 10.0,
                 pr: float = 1e-6):
        """
        初始化电力系统环境（通用化版本）
        Args:
            bus_data: Bus矩阵数据
            branch_data: 支路数据（二维数组）
            voltage_lower: 电压下限
            voltage_upper: 电压上限
            key_nodes: 关键节点索引列表
            tunable_q_nodes: 可调节无功节点配置
            ub: 基准电压（从训练样本读取）
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
        self.reset()  # 现在UB已初始化，reset不会报错
    
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
            reward = -100
            done = True
            next_state = self.state
            info = {"loss_rate": 100, "voltages": np.ones(self.Bus.shape[0]) * self.UB}
        else:
            reward = self._calculate_improved_reward(loss_rate, voltages, actual_action)
            done = False
            next_state = self._build_state(voltages)
            self.state = next_state
            self.previous_loss_rate = loss_rate
            self.previous_voltages = voltages.copy()
            info = {"loss_rate": loss_rate, "voltages": voltages}
        
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
    
    def _calculate_improved_reward(self, loss_rate, voltages, action):
        """计算改进的奖励函数"""
        base_reward = -loss_rate * 90
        
        voltage_penalty = 0
        voltage_reward = 0
        optimal_voltage = (self.v_min + self.v_max) / 2
        
        for v in voltages:
            if v < self.v_min:
                voltage_penalty += 20 * (self.v_min - v)
            elif v > self.v_max:
                voltage_penalty += 20 * (v - self.v_max)
            else:
                voltage_reward += 2
        
        improvement_bonus = 0
        if self.previous_loss_rate is not None:
            improvement = self.previous_loss_rate - loss_rate
            if improvement > 0:
                improvement_bonus = 10 * improvement
        
        constraint_bonus = 0
        if voltage_penalty == 0 and loss_rate < 5.0:
            constraint_bonus = 15
        
        total_reward = (base_reward + voltage_reward + improvement_bonus + 
                       constraint_bonus - voltage_penalty)
        
        return total_reward
    
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

# -------------------------- 训练函数（通用化版本） --------------------------
def train_rl_model():
    """通用化的RL模型训练函数"""
    # -------------------------- 1. 读取所有配置和数据 --------------------------
    print("=== 读取配置文件 ===")
    paths = get_project_paths()
    
    # 读取基础配置
    voltage_lower, voltage_upper = read_voltage_limits(paths["volcst_file"])
    key_nodes = read_key_nodes(paths["kvnd_file"])
    branch_data = read_branch_data(paths["branch_file"])  # 现在是二维数组
    
    # 读取训练样本（Bus数据和UB）
    bus_data, ub = read_training_sample(paths["hisdata_dir"])
    
    # 读取并计算可调无功节点
    tunable_q_nodes = read_tunable_q_nodes(paths["pv_file"], bus_data)
    
    # 打印配置信息
    print(f"电压上下限: [{voltage_lower:.2f}kV, {voltage_upper:.2f}kV]")
    print(f"关键节点索引: {key_nodes} (共{len(key_nodes)}个)")
    print(f"可调节无功节点: {[node[3] for node in tunable_q_nodes]} (共{len(tunable_q_nodes)}个)")
    print(f"基准电压UB: {ub:.2f}kV")
    print(f"Bus节点数: {bus_data.shape[0]}")
    print(f"支路数: {branch_data.shape[0]}")
    
    # -------------------------- 2. 初始化环境和智能体 --------------------------
    print("\n=== 初始化环境和智能体 ===")
    # 【修改点3】初始化环境时直接传入UB，避免None
    env = ImprovedPowerSystemEnv(
        bus_data=bus_data,
        branch_data=branch_data,
        voltage_lower=voltage_lower,
        voltage_upper=voltage_upper,
        key_nodes=key_nodes,
        tunable_q_nodes=tunable_q_nodes,
        ub=ub  # 直接传入基准电压
    )
    
    agent = TD3(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action
    )
    
    # -------------------------- 3. 训练参数配置 --------------------------
    max_episodes = 800
    max_steps = 3
    rewards_history = []
    loss_rates_history = []
    best_loss_rate = float('inf')
    
    print("\n=== 开始训练TD3模型 ===")
    print(f"状态维度: {env.state_dim}, 动作维度: {env.action_dim}")
    print(f"训练轮次: {max_episodes}, 每轮步数: {max_steps}")
    
    # -------------------------- 4. 预填充经验池 --------------------------
    print("预填充经验池...")
    prefill_steps = 0
    while len(agent.replay_buffer) < 5000 and prefill_steps < 10000:
        state = env.reset()
        for step in range(max_steps):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            prefill_steps += 1
            if done:
                break
            state = next_state
    print(f"预填充完成，经验池大小: {len(agent.replay_buffer)}")
    
    # -------------------------- 5. 训练循环 --------------------------
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss_rate = 0
        
        for step in range(max_steps):
            # 递减探索噪声
            exploration_noise = max(0.05, 0.3 * (1 - episode / max_episodes))
            action = agent.select_action(state, exploration_noise)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # 训练智能体（前20轮不训练）
            if episode > 20:
                agent.train()
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            episode_loss_rate = info['loss_rate'] if info['loss_rate'] else 100
            
            if done:
                break
        
        # 记录历史数据
        rewards_history.append(episode_reward)
        loss_rates_history.append(episode_loss_rate)
        
        # 保存最优模型（删除旧模型，只保留最新最优）
        best_loss_rate = save_best_model(
            agent=agent,
            filename="M1",
            best_loss_rate=best_loss_rate,
            current_loss_rate=episode_loss_rate,
            paths=paths
        )
        
        # 打印训练进度
        if episode % 50 == 0:
            print(f"Episode {episode:3d} | 奖励: {episode_reward:6.2f} | 网损率: {episode_loss_rate:.4f}% | 最优网损: {best_loss_rate:.4f}%")
    
    # -------------------------- 6. 绘制训练曲线 --------------------------
    plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.7)
    plt.xlabel('训练轮次')
    plt.ylabel('奖励')
    plt.title('训练奖励曲线')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_rates_history, alpha=0.7)
    plt.xlabel('训练轮次')
    plt.ylabel('网损率 (%)')
    plt.title('训练网损率变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # -------------------------- 7. 训练完成 --------------------------
    print("\n=== 训练完成 ===")
    print(f"最终最优网损率: {best_loss_rate:.4f}%")
    print(f"仅保留最优模型文件，非最优模型已自动删除")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        train_rl_model()
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise