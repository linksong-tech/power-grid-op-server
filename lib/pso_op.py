import numpy as np
import copy

def pso_op(Bus, Branch, tunable_q_nodes, num_particles=30, max_iter=50, 
           w=0.8, c1=1.5, c2=1.5, v_min=0.95, v_max=1.05, 
           SB=10, UB=10.38, pr=1e-6):
    """
    基于粒子群算法的电力系统无功优化函数
    
    参数:
    Bus: 节点数据数组 [节点号, 负荷有功(MW), 负荷无功(Mvar)]
    Branch: 支路数据数组 [支路号, 首节点, 尾节点, 电阻(Ω), 电抗(Ω)]
    tunable_q_nodes: 可调节点配置列表 [(节点索引, 无功最小值, 无功最大值, 节点名称), ...]
    num_particles: 粒子数量
    max_iter: 最大迭代次数
    w: 初始惯性权重
    c1: 认知系数
    c2: 社会系数
    v_min: 电压下限 (标幺值)
    v_max: 电压上限 (标幺值)
    SB: 基准功率 (MVA)
    UB: 基准电压 (kV)
    pr: 潮流收敛精度
    
    返回:
    dict: 包含优化结果的字典
    """
    
    def power_flow(tunable_q_values):
        """
        潮流计算函数
        tunable_q_values: 可调节点的无功值列表
        """
        # 复制原始数据
        Bus_copy = copy.deepcopy(Bus)
        Branch_copy = copy.deepcopy(Branch)
        
        # 更新可调节点的无功值
        for i, (node_idx, _, _, _) in enumerate(tunable_q_nodes):
            Bus_copy[node_idx, 2] = tunable_q_values[i]  # 第2列是无功
        
        # 规范化支路编号，确保为 1..branchnum，避免外部数据越界
        branchnum = Branch_copy.shape[0]
        normalized_branch_ids = np.arange(1, branchnum + 1)
        Branch_copy[:, 0] = normalized_branch_ids

        # 标幺化处理
        Bus_copy[:, 1] = Bus_copy[:, 1] / SB  # 有功标幺值
        Bus_copy[:, 2] = Bus_copy[:, 2] / SB  # 无功标幺值
        Branch_copy[:, 3] = Branch_copy[:, 3] * SB / (UB **2)  # 电阻标幺值
        Branch_copy[:, 4] = Branch_copy[:, 4] * SB / (UB** 2)  # 电抗标幺值
        
        # 节点和支路数量
        busnum = Bus_copy.shape[0]
        branchnum = Branch_copy.shape[0]
        
        # 节点类型标识
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
        Vbus[0] = 1.0  # 平衡节点电压
        cita = np.zeros(busnum)
        
        # 初始化变量
        k = 0
        Ploss = np.zeros(branchnum)
        Qloss = np.zeros(branchnum)
        P = np.zeros(branchnum)
        Q = np.zeros(branchnum)
        F = 0
        
        # 支路排序（从末端到首端）
        Branchcols = 5
        TempBranch = Branch_copy.copy()
        s1 = np.zeros((0, Branchcols))
        while TempBranch.size > 0:
            s = TempBranch.shape[0] - 1
            s2 = np.zeros((0, Branchcols))
            while s >= 0:
                i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
                if i.size == 0:
                    s1 = np.vstack([s1, TempBranch[s, :]])
                else:
                    if s2.size > 0:
                        s2 = np.vstack([s2, TempBranch[s, :]])
                    else:
                        s2 = TempBranch[s, :].reshape(1, -1)
                s -= 1
            TempBranch = s2.copy()
        
        # 潮流迭代计算
        while k < 100 and F == 0:
            Pij1 = np.zeros(busnum)
            Qij1 = np.zeros(busnum)
            
            # 前推过程
            for s in range(branchnum):
                ii = int(s1[s, 1] - 1)
                jj = int(s1[s, 2] - 1)
                Pload = Bus_copy[jj, 1]
                Qload = Bus_copy[jj, 2]
                R = s1[s, 3]
                X = s1[s, 4]
                VV = Vbus[jj]
                
                Pij0 = Pij1[jj]
                Qij0 = Qij1[jj]
                
                II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
                Ploss[int(s1[s, 0]) - 1] = II * R
                Qloss[int(s1[s, 0]) - 1] = II * X
                
                P[int(s1[s, 0]) - 1] = Pload + Ploss[int(s1[s, 0]) - 1] + Pij0
                Q[int(s1[s, 0]) - 1] = Qload + Qloss[int(s1[s, 0]) - 1] + Qij0
                
                Pij1[ii] += P[int(s1[s, 0]) - 1]
                Qij1[ii] += Q[int(s1[s, 0]) - 1]
            
            # 回推过程
            for s in range(branchnum-1, -1, -1):
                ii = int(s1[s, 2] - 1)
                kk = int(s1[s, 1] - 1)
                R = s1[s, 3]
                X = s1[s, 4]
                
                V_real = Vbus[kk] - (P[int(s1[s, 0]) - 1]*R + Q[int(s1[s, 0]) - 1]*X) / Vbus[kk]
                V_imag = (P[int(s1[s, 0]) - 1]*X - Q[int(s1[s, 0]) - 1]*R) / Vbus[kk]
                
                Vbus[ii] = np.sqrt(V_real**2 + V_imag**2)
                cita[ii] = cita[kk] - np.arctan2(V_imag, V_real)
            
            # 收敛判断
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
            if ddp < pr and ddq < pr:
                F = 1
            k += 1
        
        if k == 100:
            return None, None, None  # 不收敛
        
        # 网损率计算
        P1 = np.sum(Ploss)
        balance_node_output = Pij2[0] * SB
        pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
        pv_total_injection = sum(-Bus_copy[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * SB
        total_input_power = balance_node_output + pv_total_injection
        
        load_nodes_mask = [typ == "负荷节点" for typ in node_types]
        total_output_power = sum(Bus_copy[i, 1] for i in range(busnum) if load_nodes_mask[i]) * SB
        
        loss_rate = (total_input_power - total_output_power) / total_input_power * 100 if total_input_power != 0 else 0.0
        Vbus_kv = Vbus * UB  # 电压有名值（kV）
        
        return loss_rate, Vbus_kv, (balance_node_output, pv_total_injection, total_input_power, total_output_power)
    
    # 从配置中提取可调节点参数
    dim = len(tunable_q_nodes)  # 决策变量维度
    q_mins = np.array([node[1] for node in tunable_q_nodes])  # 各节点最小值
    q_maxs = np.array([node[2] for node in tunable_q_nodes])  # 各节点最大值
    
    # 电压约束（转换为有名值）
    v_min_kv = v_min * 10
    v_max_kv = v_max * 10
    
    # 初始化粒子群
    particles = np.random.rand(num_particles, dim)
    for i in range(dim):
        # 映射到各节点的无功范围
        particles[:, i] = particles[:, i] * (q_maxs[i] - q_mins[i]) + q_mins[i]
    
    velocities = np.zeros((num_particles, dim))
    
    # 初始化个体最优和全局最优
    pbest = np.copy(particles)
    pbest_fitness = np.ones(num_particles) * np.inf
    
    # 计算初始适应度
    for i in range(num_particles):
        loss_rate, voltages, _ = power_flow(particles[i])
        if loss_rate is None or voltages is None:
            pbest_fitness[i] = np.inf
            continue
        
        # 电压越限惩罚
        voltage_violation = np.sum(np.maximum(v_min_kv - voltages, 0) + np.maximum(voltages - v_max_kv, 0))
        pbest_fitness[i] = loss_rate + 100 * voltage_violation if voltage_violation > 0 else loss_rate
    
    # 全局最优初始化
    gbest_idx = np.argmin(pbest_fitness)
    gbest = np.copy(pbest[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    
    # 记录优化过程
    fitness_history = []
    
    # 迭代优化
    for iter in range(max_iter):
        current_w = w - (w - 0.4) * (iter / max_iter)  # 惯性权重递减
        
        for i in range(num_particles):
            # 速度更新（每个粒子独立的随机数）
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = current_w * velocities[i] + \
                           c1 * r1 * (pbest[i] - particles[i]) + \
                           c2 * r2 * (gbest - particles[i])
            
            # 速度限制
            max_vel = 0.1 * (q_maxs - q_mins)
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            # 位置更新
            particles[i] += velocities[i]
            # 位置限制
            for j in range(dim):
                particles[i, j] = np.clip(particles[i, j], q_mins[j], q_maxs[j])
            
            # 计算当前适应度
            loss_rate, voltages, _ = power_flow(particles[i])
            if loss_rate is None or voltages is None:
                current_fitness = np.inf
            else:
                voltage_violation = np.sum(np.maximum(v_min_kv - voltages, 0) + np.maximum(voltages - v_max_kv, 0))
                current_fitness = loss_rate + 100 * voltage_violation if voltage_violation > 0 else loss_rate
            
            # 更新个体最优
            if current_fitness < pbest_fitness[i]:
                pbest[i] = np.copy(particles[i])
                pbest_fitness[i] = current_fitness
        
        # 更新全局最优
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest = np.copy(pbest[current_best_idx])
            gbest_fitness = pbest_fitness[current_best_idx]
        
        # 记录历史
        fitness_history.append(np.min(pbest_fitness))
        
        # 打印迭代信息
        if (iter + 1) % 5 == 0:
            print(f"迭代 {iter + 1}/{max_iter}, 最优网损率: {fitness_history[-1]:.4f}%")

              # 计算初始状态
    initial_q_values = [Bus[node[0], 2] for node in tunable_q_nodes]
    print(f"initial_q_values", initial_q_values, Bus)
    initial_loss_rate, initial_voltages, _ = power_flow(initial_q_values)
    if initial_loss_rate is None:
        return {
            'error': '初始潮流计算不收敛',
            'convergence': False,
            'optimal_params': gbest.tolist(),
            'fitness_history': fitness_history
        }
    
    # 计算最终结果
    final_loss_rate, final_voltages, power_info = power_flow(gbest)
    if final_loss_rate is None:
        return {
            'error': '优化后潮流计算不收敛',
            'convergence': False,
            'optimal_params': gbest.tolist(),
            'fitness_history': fitness_history
        }
    
    # 构建返回结果
    result = {
        'optimal_params': gbest.tolist(),
        'optimal_loss_rate': final_loss_rate,
        'initial_loss_rate': initial_loss_rate,
        'loss_reduction': initial_loss_rate - final_loss_rate if final_loss_rate is not None else 0,
        'loss_reduction_percent': ((initial_loss_rate - final_loss_rate)/initial_loss_rate)*100 if final_loss_rate is not None and initial_loss_rate != 0 else 0,
        'fitness_history': fitness_history,
        'initial_q_values': initial_q_values,
        'node_names': [node[3] for node in tunable_q_nodes],
        'final_voltages': final_voltages.tolist() if final_voltages is not None else None,
        'initial_voltages': initial_voltages.tolist() if initial_voltages is not None else None,
        'convergence': final_loss_rate is not None,
        'power_info': {
            'balance_node_output': power_info[0],
            'pv_total_injection': power_info[1], 
            'total_input_power': power_info[2],
            'total_output_power': power_info[3]
        }
    }
    
    return result
