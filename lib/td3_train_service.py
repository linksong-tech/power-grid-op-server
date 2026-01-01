#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3训练服务 - 工程化API
提供标准化的训练接口
"""
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
import datetime
import os
import glob
import uuid
from typing import Dict, List, Tuple, Optional, Callable
from td3_core import ActorNetwork, CriticNetwork, power_flow_calculation


class TD3TrainService:
    """TD3训练服务类"""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        """
        初始化TD3训练服务
        
        Args:
            state_dim: 状态维度（关键节点数量）
            action_dim: 动作维度（可调无功节点数量）
            max_action: 最大动作值
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 复制目标网络权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 经验回放
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 128
        
        # TD3参数
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, exploration_noise: float = 0.1) -> np.ndarray:
        """选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if exploration_noise != 0:
            noise = np.random.normal(0, exploration_noise, size=self.action_dim)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action
    
    def train_step(self):
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.total_it += 1
        
        # 采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        action_batch = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device).unsqueeze(1)
        
        # 更新Critic
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * 0.99 * target_Q
        
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss = torch.nn.MSELoss()(current_Q1, target_Q) + torch.nn.MSELoss()(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 延迟更新Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)


def save_best_model(agent: TD3TrainService, model_save_dir: str, filename: str, 
                    best_loss_rate: float, current_loss_rate: float, training_id: str) -> float:
    """
    保存最优模型（同一次训练只保留一个最优模型，不同训练的模型都保留）
    
    Args:
        agent: TD3训练服务实例
        model_save_dir: 模型保存目录
        filename: 模型基础名称（如 "M1"）
        best_loss_rate: 当前最优网损率
        current_loss_rate: 当前轮次网损率
        training_id: 训练ID（用于区分不同训练）
    
    Returns:
        更新后的最优网损率
    """
    if current_loss_rate < best_loss_rate and current_loss_rate < 15:
        # 删除同一次训练的旧模型（通过 training_id 识别）
        old_models = glob.glob(os.path.join(model_save_dir, f"{filename}_{training_id}_*.pth"))
        for old_model in old_models:
            try:
                os.remove(old_model)
                print(f"删除同次训练的旧模型: {os.path.basename(old_model)}")
            except Exception as e:
                print(f"删除旧模型失败: {old_model}, 错误: {e}")
        
        # 生成新的模型文件名（格式：M1_训练ID_MMDD_HHMMSS.pth）
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m%d_%H%M%S")
        save_filename = f"{filename}_{training_id}_{time_str}.pth"
        save_path = os.path.join(model_save_dir, save_filename)
        
        # 保存新的最优模型
        agent.save_model(save_path)
        best_loss_rate = current_loss_rate
        print(f"保存最优模型: {save_filename}，网损率: {best_loss_rate:.4f}%")
    
    return best_loss_rate


def train_td3_model(
    bus_data: np.ndarray,
    branch_data: np.ndarray,
    voltage_limits: Tuple[float, float],
    key_nodes: List[int],
    tunable_q_nodes: List[Tuple[int, float, float, str]],
    ub: float,
    sb: float = 10.0,
    max_episodes: int = 800,
    max_steps: int = 3,
    model_save_path: str = "td3_model.pth",
    pr: float = 1e-6,
    training_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    训练TD3模型的主函数
    
    Args:
        bus_data: 节点数据 (n×3)
        branch_data: 支路数据 (m×5)
        voltage_limits: 电压上下限 (v_min, v_max)
        key_nodes: 关键节点索引列表
        tunable_q_nodes: 可调无功节点配置
        ub: 基准电压 kV
        sb: 基准功率 MVA
        max_episodes: 最大训练轮次
        max_steps: 每轮最大步数
        model_save_path: 模型保存路径
        pr: 潮流收敛精度（默认 1e-6）
        training_id: 训练ID（用于区分不同训练，默认自动生成）
        progress_callback: 进度回调函数 callback(episode, reward, loss_rate)
    
    Returns:
        dict: {
            'success': bool,
            'best_loss_rate': float,
            'model_path': str,
            'training_history': {
                'rewards': List[float],
                'loss_rates': List[float]
            }
        }
    """
    # 生成训练ID（如果未提供）
    if training_id is None:
        training_id = str(uuid.uuid4())[:8]  # 使用UUID的前8位，简短且唯一
    v_min, v_max = voltage_limits
    state_dim = len(key_nodes)
    action_dim = len(tunable_q_nodes)
    
    # 初始化训练服务
    agent = TD3TrainService(state_dim, action_dim)
    
    # 训练环境
    env = PowerSystemEnv(
        bus_data, branch_data, voltage_limits, key_nodes, 
        tunable_q_nodes, ub, sb, pr
    )
    
    # 预填充经验池
    prefill_steps = 0
    print(f"预填充经验池...")
    while len(agent.replay_buffer) < 5000 and prefill_steps < 10000:
        state = env.reset()
        for step in range(max_steps):
            action = np.random.uniform(-1, 1, size=action_dim)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            prefill_steps += 1
            if done:
                break
            state = next_state
    print(f"预填充完成，经验池大小: {len(agent.replay_buffer)}")
    
    # 训练循环
    rewards_history = []
    loss_rates_history = []
    best_loss_rate = float('inf')
    
    print(f"\n=== 开始训练TD3模型 ===")
    print(f"训练ID: {training_id}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"训练轮次: {max_episodes}, 每轮步数: {max_steps}")
    print(f"潮流收敛精度: {pr}")
    print(f"模型保存目录: {os.path.dirname(model_save_path)}")
    print(f"模型文件前缀: M1")
    print("")
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss_rate = 0
        
        for step in range(max_steps):
            exploration_noise = max(0.05, 0.3 * (1 - episode / max_episodes))
            action = agent.select_action(state, exploration_noise)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            if episode > 20:
                agent.train_step()
            
            state = next_state
            episode_reward += reward
            episode_loss_rate = info['loss_rate'] if info['loss_rate'] else 100
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        loss_rates_history.append(episode_loss_rate)
        
        # 保存最优模型（使用 save_best_model 函数，参考 td3onesample.py）
        # 使用 "M1" 作为默认文件名前缀（与 td3onesample.py 保持一致）
        model_save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else os.getcwd()
        filename_prefix = "M1"  # 默认使用 M1 前缀
        
        best_loss_rate = save_best_model(
            agent=agent,
            model_save_dir=model_save_dir,
            filename=filename_prefix,
            best_loss_rate=best_loss_rate,
            current_loss_rate=episode_loss_rate,
            training_id=training_id
        )
        
        # 打印训练进度（每 50 个 episode）
        if episode % 50 == 0:
            print(f"Episode {episode:3d} | 奖励: {episode_reward:6.2f} | 网损率: {episode_loss_rate:.4f}% | 最优网损: {best_loss_rate:.4f}%")
        
        # 进度回调（episode 从 0 开始，显示时 +1）
        if progress_callback:
            progress_callback(episode + 1, episode_reward, episode_loss_rate)
    
    # 获取最终保存的模型路径
    final_model_path = None
    if best_loss_rate < 15:
        model_save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else os.getcwd()
        # 查找当前训练的模型文件（通过 training_id 识别）
        model_files = glob.glob(os.path.join(model_save_dir, f"M1_{training_id}_*.pth"))
        if model_files:
            # 按修改时间排序，获取最新的
            model_files.sort(key=os.path.getmtime, reverse=True)
            final_model_path = model_files[0]
    
    print(f"\n=== 训练完成 ===")
    print(f"最终最优网损率: {best_loss_rate:.4f}%")
    if final_model_path:
        print(f"最优模型已保存: {final_model_path}")
    else:
        print(f"未达到保存条件（网损率 >= 15%），未保存模型")
    
    return {
        'success': True,
        'best_loss_rate': best_loss_rate,
        'model_path': final_model_path,
        'training_history': {
            'rewards': rewards_history,
            'loss_rates': loss_rates_history
        }
    }


class PowerSystemEnv:
    """电力系统环境（简化版）"""
    def __init__(self, bus_data, branch_data, voltage_limits, key_nodes, 
                 tunable_q_nodes, ub, sb, pr=1e-6):
        self.Bus = bus_data
        self.Branch = branch_data
        self.v_min, self.v_max = voltage_limits
        self.key_nodes = key_nodes
        self.tunable_q_nodes = tunable_q_nodes
        self.ub = ub
        self.sb = sb
        self.pr = pr  # 潮流收敛精度
        self.q_mins = np.array([node[1] for node in tunable_q_nodes])
        self.q_maxs = np.array([node[2] for node in tunable_q_nodes])
        self.previous_loss_rate = None
        self.previous_voltages = None  # 保存上一次的电压值
    
    def reset(self):
        initial_q = [self.Bus[node[0], 2] for node in self.tunable_q_nodes]
        _, initial_voltages, _ = power_flow_calculation(
            self.Bus, self.Branch, self.tunable_q_nodes, initial_q, self.sb, self.ub, self.pr
        )
        if initial_voltages is None:
            initial_voltages = np.ones(self.Bus.shape[0]) * self.ub
        self.state = self._build_state(initial_voltages)
        self.previous_loss_rate = None
        self.previous_voltages = initial_voltages.copy() if initial_voltages is not None else None
        return self.state
    
    def _build_state(self, voltages):
        key_voltages = voltages[self.key_nodes]
        normalized = (key_voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        return normalized
    
    def step(self, action):
        actual_action = self._denormalize_action(action)
        loss_rate, voltages, _ = power_flow_calculation(
            self.Bus, self.Branch, self.tunable_q_nodes, actual_action, self.sb, self.ub, self.pr
        )
        
        if loss_rate is None:
            return self.state, -100, True, {"loss_rate": 100}
        
        reward = self._calculate_reward(loss_rate, voltages, action)
        next_state = self._build_state(voltages)
        self.state = next_state
        self.previous_loss_rate = loss_rate
        self.previous_voltages = voltages.copy()
        
        return next_state, reward, False, {"loss_rate": loss_rate, "voltages": voltages}
    
    def _denormalize_action(self, action):
        actual_actions = []
        for i in range(len(action)):
            normalized = np.clip(action[i], -1, 1)
            actual = (normalized + 1) / 2 * (self.q_maxs[i] - self.q_mins[i]) + self.q_mins[i]
            actual_actions.append(actual)
        return actual_actions
    
    def _calculate_reward(self, loss_rate, voltages, action=None):
        """
        计算奖励函数
        
        Args:
            loss_rate: 网损率
            voltages: 节点电压数组
            action: 动作（保留参数以保持接口一致性，当前未使用）
        """
        base_reward = -loss_rate * 90
        voltage_penalty = sum(20 * max(self.v_min - v, 0, v - self.v_max) for v in voltages)
        voltage_reward = sum(2 for v in voltages if self.v_min <= v <= self.v_max)
        improvement_bonus = 0
        if self.previous_loss_rate:
            improvement = self.previous_loss_rate - loss_rate
            if improvement > 0:
                improvement_bonus = 10 * improvement
        constraint_bonus = 15 if voltage_penalty == 0 and loss_rate < 5.0 else 0
        return base_reward + voltage_reward + improvement_bonus + constraint_bonus - voltage_penalty
