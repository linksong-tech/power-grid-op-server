#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潮流计算核心函数
"""
import numpy as np
import copy


def power_flow_calculation(bus_data, branch_data, tunable_q_nodes, tunable_q_values, 
                           sb=10.0, ub=10.38, pr=1e-6):
    """
    前推回代潮流计算
    
    Args:
        bus_data: 节点数据 (n×3) [节点号, 有功P, 无功Q]
        branch_data: 支路数据 (m×5) [线路号, 首节点, 末节点, 电阻R, 电抗X]
        tunable_q_nodes: 可调无功节点配置 [(节点索引, Q_min, Q_max, 节点名), ...]
        tunable_q_values: 可调无功值列表
        sb: 基准功率 MVA
        ub: 基准电压 kV
        pr: 收敛精度
    
    Returns:
        tuple: (网损率%, 节点电压kV数组, 功率信息元组) 或 (None, None, None)
    """
    Bus_copy = copy.deepcopy(bus_data)
    Branch_copy = copy.deepcopy(branch_data)
    
    # 更新可调无功节点的Q值
    for i, (node_idx, _, _, _) in enumerate(tunable_q_nodes):
        if i < len(tunable_q_values):
            Bus_copy[node_idx, 2] = tunable_q_values[i]
    
    # 标幺化
    Bus_copy[:, 1] = Bus_copy[:, 1] / sb
    Bus_copy[:, 2] = Bus_copy[:, 2] / sb
    Branch_copy[:, 3] = Branch_copy[:, 3] * sb / (ub ** 2)
    Branch_copy[:, 4] = Branch_copy[:, 4] * sb / (ub ** 2)
    
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
    
    # 初始化
    Vbus = np.ones(busnum)
    Vbus[0] = 1.0
    cita = np.zeros(busnum)
    
    Ploss = np.zeros(branchnum)
    Qloss = np.zeros(branchnum)
    P = np.zeros(branchnum)
    Q = np.zeros(branchnum)
    
    # 支路排序
    TempBranch = Branch_copy.copy()
    s1 = np.zeros((0, 5))
    while TempBranch.size > 0:
        s = TempBranch.shape[0] - 1
        s2 = np.zeros((0, 5))
        while s >= 0:
            i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
            if i.size == 0:
                if s1.size == 0:
                    s1 = TempBranch[s, :].reshape(1, -1)
                else:
                    s1 = np.vstack([s1, TempBranch[s, :]])
            else:
                if s2.size == 0:
                    s2 = TempBranch[s, :].reshape(1, -1)
                else:
                    s2 = np.vstack([s2, TempBranch[s, :]])
            s -= 1
        TempBranch = s2.copy()
    
    # 迭代计算
    k = 0
    F = 0
    while k < 100 and F == 0:
        Pij1 = np.zeros(busnum)
        Qij1 = np.zeros(busnum)
        
        # 前推
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
            
            II = ((Pload + Pij0)**2 + (Qload + Qij0)**2) / (VV**2)
            Ploss[s] = II * R
            Qloss[s] = II * X
            
            P[s] = Pload + Ploss[s] + Pij0
            Q[s] = Qload + Qloss[s] + Qij0
            
            Pij1[ii] += P[s]
            Qij1[ii] += Q[s]
        
        # 回代
        for s in range(branchnum-1, -1, -1):
            ii = int(s1[s, 2] - 1)
            kk = int(s1[s, 1] - 1)
            R = s1[s, 3]
            X = s1[s, 4]
            
            V_real = Vbus[kk] - (P[s]*R + Q[s]*X) / Vbus[kk]
            V_imag = (P[s]*X - Q[s]*R) / Vbus[kk]
            
            Vbus[ii] = np.sqrt(V_real**2 + V_imag**2)
            cita[ii] = cita[kk] - np.arctan2(V_imag, V_real)
        
        # 收敛检查
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
            
            II = ((Pload + Pij0)**2 + (Qload + Qij0)**2) / (VV**2)
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
        return None, None, None
    
    # 计算网损率
    balance_node_output = Pij2[0] * sb
    pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
    pv_total_injection = sum(-Bus_copy[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * sb
    total_input_power = balance_node_output + pv_total_injection
    
    load_nodes_mask = [typ == "负荷节点" for typ in node_types]
    total_output_power = sum(Bus_copy[i, 1] for i in range(busnum) if load_nodes_mask[i]) * sb
    
    loss_rate = (total_input_power - total_output_power) / total_input_power * 100 if total_input_power != 0 else 0.0
    Vbus_kv = Vbus * ub
    
    return loss_rate, Vbus_kv, (balance_node_output, pv_total_injection, total_input_power, total_output_power)
