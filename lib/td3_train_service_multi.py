"""
TD3训练服务 (多样本版本) - 纯NumPy版本
完全替代PyTorch实现，提供相同的训练接口，支持多样本分层训练
"""
import numpy as np
import random
import datetime
import os
import glob
import uuid
import sys
from typing import List, Tuple, Dict, Optional, Callable

# -------------------------- TD3 核心组件 --------------------------
from td3_train_service_numpy import TD3TrainServiceNumpy
from td3_core.power_flow import power_flow_calculation

# -------------------------- 改进的强化学习环境（自动计算基准网损率） --------------------------
class ImprovedPowerSystemEnv:
    def __init__(self, 
                 bus_data: np.ndarray,
                 branch_data: np.ndarray,
                 voltage_lower: float,
                 voltage_upper: float,
                 key_nodes: List[int],
                 tunable_q_nodes_config: List[Dict],
                 ub: float,
                 sb: float = 10.0,
                 pr: float = 1e-6,
                 baseline_info: Dict[str, float] = None):
        """
        初始化电力系统环境（改进版本，支持基准网损率）
        Args:
            bus_data: Bus矩阵数据
            branch_data: 支路数据（二维数组）
            voltage_lower: 电压下限
            voltage_upper: 电压上限
            key_nodes: 关键节点索引列表
            tunable_q_nodes_config: 可调无功节点配置列表 [{'node_index': int, 'capacity': float, 'node_name': str}]
            ub: 基准电压（从训练样本读取）
            sb: 基准功率 MVA
            pr: 潮流收敛精度
            baseline_info: 基准信息字典，包含baseline_loss（基准网损率）和total_power（总功率）
        """
        self.Bus = bus_data.copy()
        self.Branch = branch_data
        self.SB = sb
        self.UB = ub
        self.pr = pr
        
        self.v_min = voltage_lower
        self.v_max = voltage_upper
        self.key_nodes = key_nodes
        
        # 动态计算每个样本的无功上下限
        self.tunable_q_nodes = []
        for config in tunable_q_nodes_config:
            node_index = config['node_index']
            capacity = config['capacity']
            node_name = config['node_name']
            
            # 计算Q limit
            p_current = self.Bus[node_index, 1]
            q_max = np.sqrt(max(0, capacity**2 - p_current**2))
            q_min = -q_max
            
            # (index, min, max, name)
            self.tunable_q_nodes.append((node_index, q_min, q_max, node_name))
            
        self.state_dim = len(key_nodes)
        self.action_dim = len(self.tunable_q_nodes)
        self.max_action = 1.0
        
        self.q_mins = np.array([node[1] for node in self.tunable_q_nodes])
        self.q_maxs = np.array([node[2] for node in self.tunable_q_nodes])
        
        self.previous_loss_rate = None
        self.previous_voltages = None
        self.previous_actions = None
        
        # 基准信息
        if baseline_info is None:
            self.baseline_info = self._calculate_baseline_info()
        else:
            self.baseline_info = baseline_info
            
        # 计算归一化参数
        self._calculate_normalization_params()
        
    def _calculate_baseline_info(self) -> Dict[str, float]:
        """自动计算基准网损率和总功率"""
        # 使用初始无功值计算潮流
        initial_q = [self.Bus[node[0], 2] for node in self.tunable_q_nodes]
        loss_rate, voltages, power_info = power_flow_calculation(
            self.Bus, self.Branch, self.tunable_q_nodes, initial_q, self.SB, self.UB, self.pr
        )
        
        if loss_rate is None or power_info is None:
            # 估算
            load_power = np.sum(np.abs(self.Bus[self.Bus[:, 1] > 0, 1]))
            total_input_power = load_power * 1.1
            baseline_loss = 10.0
        else:
            total_input_power = power_info[2]
            baseline_loss = loss_rate
            
        return {
            'baseline_loss': baseline_loss,
            'total_power': total_input_power
        }

    def _calculate_normalization_params(self):
        """基于基准信息计算归一化参数"""
        baseline_loss = self.baseline_info['baseline_loss']
        total_power = self.baseline_info['total_power']
        
        # 1. 基于功率的权重
        self.power_weight = np.clip(total_power / 100.0, 0.5, 2.0)
        
        # 2. 基于基准网损率的难度系数
        if baseline_loss > 8.0:
            self.difficulty_factor = 0.8  # 容易
        elif baseline_loss > 5.0:
            self.difficulty_factor = 1.0  # 中等
        elif baseline_loss > 3.0:
            self.difficulty_factor = 1.2  # 较难
        else:
            self.difficulty_factor = 1.5  # 很难

    def reset(self):
        """重置环境"""
        initial_q = [self.Bus[node[0], 2] for node in self.tunable_q_nodes]
        _, initial_voltages, _ = power_flow_calculation(
            self.Bus, self.Branch, self.tunable_q_nodes, initial_q, self.SB, self.UB, self.pr
        )
        
        if initial_voltages is None:
            initial_voltages = np.ones(self.Bus.shape[0]) * self.UB
            
        self.state = self._build_state(initial_voltages)
        self.previous_loss_rate = None
        self.previous_voltages = initial_voltages.copy()
        self.previous_actions = None
        
        return self.state

    def _build_state(self, voltages):
        key_voltages = voltages[self.key_nodes]
        normalized = (key_voltages - self.v_min) / (self.v_max - self.v_min) * 2 - 1
        return normalized

    def step(self, action):
        """执行动作"""
        actual_action = self._denormalize_action(action)
        loss_rate, voltages, _ = power_flow_calculation(
            self.Bus, self.Branch, self.tunable_q_nodes, actual_action, self.SB, self.UB, self.pr
        )
        
        if loss_rate is None:
            # 潮流不收敛惩罚
            info = {
                "loss_rate": 100, 
                "voltages": np.ones(self.Bus.shape[0]) * self.UB,
                "normalized_improvement": -100,
                "relative_improvement": -100,
                "baseline_loss": self.baseline_info['baseline_loss']
            }
            return self.state, -200, True, info
            
        # 计算归一化奖励
        reward, normalized_improvement, relative_improvement = self._calculate_normalized_reward(
            loss_rate, voltages, actual_action
        )
        
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
        
        return next_state, reward, False, info

    def _denormalize_action(self, action):
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
        
        # 1. 相对改善百分比
        if baseline_loss > 0:
            relative_improvement = (baseline_loss - loss_rate) / baseline_loss * 100
        else:
            relative_improvement = 0
            
        # 2. 归一化改善程度
        normalized_improvement = relative_improvement * self.power_weight * self.difficulty_factor
        
        # 3. 基础奖励
        base_reward = normalized_improvement * 2.0
        
        # 4. 电压约束惩罚
        voltage_penalty = 0
        voltage_reward = 0
        for v in voltages:
            if v < self.v_min:
                voltage_penalty += 100 * (self.v_min - v)
            elif v > self.v_max:
                voltage_penalty += 100 * (v - self.v_max)
            else:
                optimal = (self.v_min + self.v_max) / 2
                distance = abs(v - optimal) / optimal
                voltage_reward += 5 * (1 - distance)
                
        # 5. 动作平滑性奖励
        action_penalty = 0
        if self.previous_actions is not None:
            # 计算动作变化量
            # 由于实际动作范围不一致，我们使用归一化空间计算变化或简单总和
            # 这里简单处理，使用L1距离
            action_change = np.sum(np.abs(np.array(actual_action) - np.array(self.previous_actions)))
            # 鼓励平滑调节，大幅变化有惩罚
            if action_change > self.max_action * 0.5:  # 变化超过50%
                action_penalty = 5 * (action_change - self.max_action * 0.5)
             
        # 6. 收敛性奖励
        convergence_bonus = 0
        if self.previous_loss_rate is not None:
            speed = (self.previous_loss_rate - loss_rate) / max(self.previous_loss_rate, 1.0)
            if speed > 0.1:
                convergence_bonus = 10 * speed
                
        # 7. 综合奖励
        total_reward = (
            base_reward + 
            voltage_reward + 
            convergence_bonus - 
            voltage_penalty - 
            action_penalty
        )
        
        # 8. 成功奖励
        if (voltage_penalty == 0 and 
            relative_improvement > 10 and 
            action_penalty < 2):
            total_reward += 20
            
        return total_reward, normalized_improvement, relative_improvement


# -------------------------- 多样本训练主逻辑 --------------------------

def train_td3_model_multisample(
    samples_data: List[Dict],      # 包含 bus_data, ub, filename
    common_config: Dict,           # 包含 branch_data, voltage_limits, key_nodes, tunable_q_nodes_config
    sb: float = 10.0,
    max_episodes_per_sample: int = 10,  # 每个样本每次epoch训练多少episode
    epochs: int = 50,                   # 总轮数
    max_steps: int = 3,
    model_save_path: str = "td3_model_multi.npz",
    pr: float = 1e-6,
    training_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    多样本分层训练主函数
    """
    if training_id is None:
        training_id = str(uuid.uuid4())[:8]

    # 解包配置
    branch_data = common_config['branch_data']
    voltage_limits = common_config['voltage_limits']
    key_nodes = common_config['key_nodes']
    tunable_q_nodes_config = common_config['tunable_q_nodes_config']
    
    state_dim = len(key_nodes)
    action_dim = len(tunable_q_nodes_config)
    
    # 初始化Agent (使用 TD3TrainServiceNumpy)
    agent = TD3TrainServiceNumpy(state_dim, action_dim, max_action=1.0)
    
    # 预计算所有样本的基准信息（提升效率）
    print("正在计算样本基准信息...")
    sample_baselines = {}
    valid_samples = []
    
    for sample in samples_data:
        try:
            temp_env = ImprovedPowerSystemEnv(
                bus_data=sample['bus_data'],
                branch_data=branch_data,
                voltage_lower=voltage_limits[0],
                voltage_upper=voltage_limits[1],
                key_nodes=key_nodes,
                tunable_q_nodes_config=tunable_q_nodes_config,
                ub=sample['ub'],
                sb=sb,
                pr=pr
            )
            sample['baseline_info'] = temp_env.baseline_info
            valid_samples.append(sample)
        except Exception as e:
            print(f"样本 {sample['filename']} 基准计算失败: {e}")
            
    if not valid_samples:
        raise ValueError("没有有效的训练样本")
        
    print(f"有效样本数: {len(valid_samples)}")
    
    # 预填充经验池
    print("预填充经验池...")
    prefill_count = 0
    # 随机选样本填充
    while len(agent.replay_buffer) < 5000 and prefill_count < 20000:
        sample = random.choice(valid_samples)
        env = ImprovedPowerSystemEnv(
            bus_data=sample['bus_data'],
            branch_data=branch_data,
            voltage_lower=voltage_limits[0],
            voltage_upper=voltage_limits[1],
            key_nodes=key_nodes,
            tunable_q_nodes_config=tunable_q_nodes_config,
            ub=sample['ub'],
            sb=sb,
            pr=pr,
            baseline_info=sample['baseline_info']
        )
        state = env.reset()
        for _ in range(max_steps):
            action = np.random.uniform(-1, 1, size=action_dim)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            prefill_count += 1
            if done: break
            state = next_state
            
    # 训练循环
    best_normalized_improvement = -float('inf')
    best_avg_loss_rate = float('inf')
    
    history = {
        'rewards': [],
        'loss_rates': [],
        'normalized_improvements': []
    }
    
    total_episodes_counter = 0
    total_max_episodes = epochs * len(valid_samples) * max_episodes_per_sample
    
    print(f"\n=== 开始多样本分层训练 ===")
    print(f"Epochs: {epochs}, Samples: {len(valid_samples)}, Episodes/Sample: {max_episodes_per_sample}")
    
    model_save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else os.getcwd()
    filename_prefix = "M1"

    for epoch in range(epochs):
        epoch_rewards = []
        epoch_improvements = []
        epoch_loss_rates = []
        
        # 遍历每个样本
        for i, sample in enumerate(valid_samples):
            env = ImprovedPowerSystemEnv(
                bus_data=sample['bus_data'],
                branch_data=branch_data,
                voltage_lower=voltage_limits[0],
                voltage_upper=voltage_limits[1],
                key_nodes=key_nodes,
                tunable_q_nodes_config=tunable_q_nodes_config,
                ub=sample['ub'],
                sb=sb,
                pr=pr,
                baseline_info=sample['baseline_info']
            )
            
            # 每个样本训练若干episode
            for episode_in_sample in range(max_episodes_per_sample):
                state = env.reset()
                ep_reward = 0
                ep_norm_imp = 0
                ep_loss = 100
                relative_imp = 0
                
                for step in range(max_steps):
                    # 动态噪声
                    noise = max(0.05, 0.3 * (1 - total_episodes_counter / total_max_episodes))
                    action = agent.select_action(state, noise)
                    next_state, reward, done, info = env.step(action)
                    
                    agent.replay_buffer.append((state, action, reward, next_state, done))
                    if len(agent.replay_buffer) > agent.batch_size:
                        # 训练步骤：调用 TD3TrainServiceNumpy 的 train_step
                        agent.train_step()
                        
                    state = next_state
                    ep_reward += reward
                    ep_norm_imp = info.get('normalized_improvement', 0)
                    ep_loss = info.get('loss_rate', 100)
                    relative_imp = info.get('relative_improvement', 0)
                    
                    if done: break
                    
                total_episodes_counter += 1
                epoch_rewards.append(ep_reward)
                epoch_improvements.append(ep_norm_imp)
                epoch_loss_rates.append(ep_loss)
                
                # 回调
                if progress_callback:
                    progress_data = {
                        'total_episode': total_episodes_counter,
                        'reward': ep_reward,
                        'loss_rate': ep_loss,
                        'normalized_improvement': ep_norm_imp,
                        'epoch': epoch + 1,
                        'total_epochs': epochs,
                        'sample_idx': i + 1,
                        'total_samples': len(valid_samples),
                        'sample_name': sample['filename'],
                        'episode_in_sample': episode_in_sample + 1,
                        'episodes_per_sample': max_episodes_per_sample,
                        'baseline_loss': sample['baseline_info']['baseline_loss'],
                        'relative_improvement': relative_imp
                    }
                    progress_callback(progress_data)
                    
        # Epoch 结束统计
        avg_reward = np.mean(epoch_rewards)
        avg_imp = np.mean(epoch_improvements)
        avg_loss = np.mean(epoch_loss_rates)
        
        history['rewards'].append(avg_reward)
        history['normalized_improvements'].append(avg_imp)
        history['loss_rates'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {avg_reward:.2f} | Avg Norm Imp: {avg_imp:.2f}% | Avg Loss: {avg_loss:.4f}%")
        
        # 保存最优模型 (基于归一化改善程度)
        if avg_imp > best_normalized_improvement:
            try:
                # 查找并删除同次训练的旧模型
                old_pattern = os.path.join(model_save_dir, f"{filename_prefix}_{training_id}_*.npz")
                old_models = glob.glob(old_pattern)
                for old_model in old_models:
                    try:
                        os.remove(old_model)
                        print(f"删除旧模型: {os.path.basename(old_model)}")
                    except Exception as e:
                        print(f"删除旧模型失败: {e}")
            except Exception as e:
                print(f"清理旧模型出错: {e}")
                
            timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
            save_name = f"{filename_prefix}_{training_id}_{timestamp}.npz"
            final_path = os.path.join(model_save_dir, save_name)
            
            # 使用 TD3TrainServiceNumpy 的 save_model
            agent.save_model(final_path)
            
            best_normalized_improvement = avg_imp
            best_avg_loss_rate = avg_loss
            print(f"新最优模型保存! 改善: {avg_imp:.2f}%")

    print("\n训练结束.")
    return {
        'success': True,
        'best_normalized_improvement': best_normalized_improvement,
        'best_avg_loss_rate': best_avg_loss_rate,
        'history': history
    }
