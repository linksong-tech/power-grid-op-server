#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3训练服务 - 纯NumPy版本
完全替代PyTorch实现，提供相同的训练接口
"""
import numpy as np
from collections import deque
import random
import datetime
import os
import glob
import uuid
from typing import Dict, List, Tuple, Optional, Callable

from td3_core.numpy_models import ActorNetworkNumpy, CriticNetworkNumpy
from td3_core.numpy_backend import (
    AdamOptimizer, clip_grad_norm, soft_update, copy_params,
    mse_loss_gradient, clip_array, DTYPE
)
from td3_core.power_flow import power_flow_calculation


class TD3TrainServiceNumpy:
    """TD3训练服务类 - 纯NumPy实现"""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        """
        初始化TD3训练服务
        
        Args:
            state_dim: 状态维度（关键节点数量）
            action_dim: 动作维度（可调无功节点数量）
            max_action: 最大动作值
        """
        print(f"使用设备: CPU (NumPy {DTYPE.__name__})")
        
        # 初始化主网络
        self.actor = ActorNetworkNumpy(state_dim, action_dim, max_action)
        self.critic = CriticNetworkNumpy(state_dim, action_dim)
        
        # 初始化目标网络并复制参数
        self.actor_target = ActorNetworkNumpy(state_dim, action_dim, max_action)
        self.actor_target.set_params(self.actor.get_params())
        
        self.critic_target = CriticNetworkNumpy(state_dim, action_dim)
        critic_params1, critic_params2 = self.critic.get_params()
        self.critic_target.set_params(critic_params1, critic_params2)
        
        # 初始化优化器
        self.actor_optimizer = AdamOptimizer(self.actor.get_params(), lr=1e-4)
        
        # Critic优化器需要扁平化参数
        critic_flat_params = self._flatten_critic_params()
        self.critic_optimizer = AdamOptimizer(critic_flat_params, lr=1e-3)
        
        # 超参数
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 128
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
    
    def _flatten_critic_params(self) -> Dict[str, np.ndarray]:
        """扁平化Critic参数（用于优化器）"""
        params1, params2 = self.critic.get_params()
        flat_params = {}
        for k, v in params1.items():
            flat_params[f"c1_{k}"] = v
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
        """
        选择动作
        
        Args:
            state: 状态输入
            exploration_noise: 探索噪声标准差
        
        Returns:
            动作数组
        """
        state = state.reshape(1, -1).astype(DTYPE)
        action = self.actor.forward(state, save_cache=False).flatten()
        
        if exploration_noise != 0:
            noise = np.random.normal(0, exploration_noise, size=self.action_dim).astype(DTYPE)
            noise = clip_array(noise, -self.noise_clip, self.noise_clip)
            action = clip_array(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def _compute_critic_loss_and_grads(self, state_batch: np.ndarray, action_batch: np.ndarray, 
                                       target_Q: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        计算Critic损失和梯度
        
        Args:
            state_batch: 状态批次
            action_batch: 动作批次
            target_Q: 目标Q值
        
        Returns:
            (损失值, 梯度字典)
        """
        # 前向传播（保存缓存用于反向传播）
        current_Q1, current_Q2 = self.critic.forward(state_batch, action_batch, save_cache=True)
        
        # 计算MSE损失
        loss1 = np.mean(np.square(current_Q1 - target_Q))
        loss2 = np.mean(np.square(current_Q2 - target_Q))
        total_loss = loss1 + loss2
        
        # 计算梯度
        grad_q1 = mse_loss_gradient(current_Q1, target_Q, self.batch_size)
        grad_q2 = mse_loss_gradient(current_Q2, target_Q, self.batch_size)
        
        # 反向传播
        grads1, grads2 = self.critic.backward(grad_q1, grad_q2)
        
        # 扁平化梯度
        flat_grads = {}
        for k, v in grads1.items():
            flat_grads[f"c1_{k}"] = v
        for k, v in grads2.items():
            flat_grads[f"c2_{k}"] = v
        
        return total_loss, flat_grads
    
    def _compute_actor_loss_and_grads(self, state_batch: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        计算Actor损失和梯度
        
        Args:
            state_batch: 状态批次
        
        Returns:
            (损失值, 梯度字典)
        """
        # Actor前向传播
        action = self.actor.forward(state_batch, save_cache=True)
        
        # 通过Critic计算Q值
        q1 = self.critic.Q1(state_batch, action)
        
        # Actor损失：-mean(Q1)（最大化Q值）
        actor_loss = -np.mean(q1)
        
        # 损失对Q1的梯度
        grad_q1 = -np.ones_like(q1, dtype=DTYPE) / self.batch_size
        
        # 通过Critic反向传播到action
        x = np.concatenate([state_batch, action], axis=1)
        
        # Critic网络1前向（保存中间结果）
        c_z1 = np.dot(x, self.critic.params1["w1"]) + self.critic.params1["b1"]
        c_h1 = np.maximum(0, c_z1).astype(DTYPE)
        c_z2 = np.dot(c_h1, self.critic.params1["w2"]) + self.critic.params1["b2"]
        c_h2 = np.maximum(0, c_z2).astype(DTYPE)
        c_z3 = np.dot(c_h2, self.critic.params1["w3"]) + self.critic.params1["b3"]
        c_h3 = np.maximum(0, c_z3).astype(DTYPE)
        
        # Critic反向传播
        grad_c_h3 = np.dot(grad_q1, self.critic.params1["w4"].T)
        grad_c_h3 = grad_c_h3 * (c_h3 > 0).astype(DTYPE)
        
        grad_c_h2 = np.dot(grad_c_h3, self.critic.params1["w3"].T)
        grad_c_h2 = grad_c_h2 * (c_h2 > 0).astype(DTYPE)
        
        grad_c_h1 = np.dot(grad_c_h2, self.critic.params1["w2"].T)
        grad_c_h1 = grad_c_h1 * (c_h1 > 0).astype(DTYPE)
        
        grad_x = np.dot(grad_c_h1, self.critic.params1["w1"].T)
        
        # 提取对action的梯度
        grad_action = grad_x[:, -self.action_dim:]
        
        # Actor反向传播
        actor_grads = self.actor.backward(grad_action)
        
        return actor_loss, actor_grads
    
    def train_step(self):
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.total_it += 1
        
        # 采样批次数据
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = np.array([e[0] for e in batch], dtype=DTYPE)
        action_batch = np.array([e[1] for e in batch], dtype=DTYPE)
        reward_batch = np.array([e[2] for e in batch], dtype=DTYPE).reshape(-1, 1)
        next_state_batch = np.array([e[3] for e in batch], dtype=DTYPE)
        done_batch = np.array([e[4] for e in batch], dtype=DTYPE).reshape(-1, 1)
        
        # 计算目标Q值
        noise = np.random.normal(0, self.policy_noise, size=action_batch.shape).astype(DTYPE)
        noise = clip_array(noise, -self.noise_clip, self.noise_clip)
        
        next_actions = self.actor_target.forward(next_state_batch, save_cache=False) + noise
        next_actions = clip_array(next_actions, -self.max_action, self.max_action)
        
        target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_actions, save_cache=False)
        target_Q = np.minimum(target_Q1, target_Q2)
        target_Q = reward_batch + (1 - done_batch) * 0.99 * target_Q
        
        # 更新Critic
        critic_loss, critic_grads = self._compute_critic_loss_and_grads(state_batch, action_batch, target_Q)
        critic_grads = clip_grad_norm(critic_grads)
        
        critic_flat_params = self._flatten_critic_params()
        updated_critic_params = self.critic_optimizer.step(critic_flat_params, critic_grads)
        
        params1, params2 = self._unflatten_critic_params(updated_critic_params)
        self.critic.set_params(params1, params2)
        
        # 延迟更新Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss, actor_grads = self._compute_actor_loss_and_grads(state_batch)
            actor_grads = clip_grad_norm(actor_grads)
            
            updated_actor_params = self.actor_optimizer.step(self.actor.get_params(), actor_grads)
            self.actor.set_params(updated_actor_params)
            
            # 软更新目标网络
            critic_params1, critic_params2 = self.critic.get_params()
            target_critic_params1, target_critic_params2 = self.critic_target.get_params()
            
            updated_target_critic_params1 = soft_update(target_critic_params1, critic_params1, tau=0.005)
            updated_target_critic_params2 = soft_update(target_critic_params2, critic_params2, tau=0.005)
            self.critic_target.set_params(updated_target_critic_params1, updated_target_critic_params2)
            
            actor_params = self.actor.get_params()
            target_actor_params = self.actor_target.get_params()
            updated_target_actor_params = soft_update(target_actor_params, actor_params, tau=0.005)
            self.actor_target.set_params(updated_target_actor_params)
    
    def save_model(self, filepath: str):
        """保存模型"""
        actor_params = self.actor.get_params()
        critic_params1, critic_params2 = self.critic.get_params()
        
        save_dict = {}
        save_dict.update({f"actor_{k}": v for k, v in actor_params.items()})
        save_dict.update({f"critic1_{k}": v for k, v in critic_params1.items()})
        save_dict.update({f"critic2_{k}": v for k, v in critic_params2.items()})
        
        np.savez(filepath, **save_dict)


def save_best_model(agent: TD3TrainServiceNumpy, model_save_dir: str, filename: str, 
                    best_loss_rate: float, current_loss_rate: float, training_id: str) -> float:
    """
    保存最优模型（同一次训练只保留一个最优模型）
    
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
        # 删除同一次训练的旧模型
        old_models = glob.glob(os.path.join(model_save_dir, f"{filename}_{training_id}_*.npz"))
        for old_model in old_models:
            try:
                os.remove(old_model)
                print(f"删除同次训练的旧模型: {os.path.basename(old_model)}")
            except Exception as e:
                print(f"删除旧模型失败: {old_model}, 错误: {e}")
        
        # 生成新的模型文件名
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m%d_%H%M%S")
        save_filename = f"{filename}_{training_id}_{time_str}.npz"
        save_path = os.path.join(model_save_dir, save_filename)
        
        # 保存新的最优模型
        agent.save_model(save_path)
        best_loss_rate = current_loss_rate
        print(f"保存最优模型: {save_filename}，网损率: {best_loss_rate:.4f}%")
    
    return best_loss_rate


class PowerSystemEnv:
    """电力系统环境"""
    def __init__(self, bus_data, branch_data, voltage_limits, key_nodes, 
                 tunable_q_nodes, ub, sb, pr=1e-6):
        self.Bus = bus_data
        self.Branch = branch_data
        self.v_min, self.v_max = voltage_limits
        self.key_nodes = key_nodes
        self.tunable_q_nodes = tunable_q_nodes
        self.ub = ub
        self.sb = sb
        self.pr = pr
        self.q_mins = np.array([node[1] for node in tunable_q_nodes])
        self.q_maxs = np.array([node[2] for node in tunable_q_nodes])
        self.previous_loss_rate = None
        self.previous_voltages = None
    
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
        
        reward = self._calculate_reward(loss_rate, voltages)
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
    
    def _calculate_reward(self, loss_rate, voltages):
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
    model_save_path: str = "td3_model.npz",
    pr: float = 1e-6,
    training_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    训练TD3模型的主函数（纯NumPy版本）
    
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
        pr: 潮流收敛精度
        training_id: 训练ID
        progress_callback: 进度回调函数
    
    Returns:
        训练结果字典
    """
    if training_id is None:
        training_id = str(uuid.uuid4())[:8]
    
    v_min, v_max = voltage_limits
    state_dim = len(key_nodes)
    action_dim = len(tunable_q_nodes)
    
    # 初始化训练服务
    agent = TD3TrainServiceNumpy(state_dim, action_dim)
    
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
    
    print(f"\n=== 开始训练TD3模型（NumPy版本）===")
    print(f"训练ID: {training_id}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"训练轮次: {max_episodes}, 每轮步数: {max_steps}")
    print(f"潮流收敛精度: {pr}")
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
        
        # 保存最优模型
        model_save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else os.getcwd()
        filename_prefix = "M1"
        
        best_loss_rate = save_best_model(
            agent=agent,
            model_save_dir=model_save_dir,
            filename=filename_prefix,
            best_loss_rate=best_loss_rate,
            current_loss_rate=episode_loss_rate,
            training_id=training_id
        )
        
        if episode % 50 == 0:
            print(f"Episode {episode:3d} | 奖励: {episode_reward:6.2f} | 网损率: {episode_loss_rate:.4f}% | 最优网损: {best_loss_rate:.4f}%")
        
        if progress_callback:
            progress_callback(episode + 1, episode_reward, episode_loss_rate)
    
    # 获取最终模型路径
    final_model_path = None
    if best_loss_rate < 15:
        model_save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else os.getcwd()
        model_files = glob.glob(os.path.join(model_save_dir, f"M1_{training_id}_*.npz"))
        if model_files:
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

