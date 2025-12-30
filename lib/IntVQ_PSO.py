#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 23:42:00 2025

@author: ryne
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# 关闭tkinter主窗口（仅用文件选择弹窗）
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)  # 弹窗置顶

plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]  # "Heiti TC"是 macOS 自带中文字体，sans-serif是后备
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# -------------------------- 文件读取通用函数 --------------------------
def select_excel_file(title="选择Excel文件"):
    """
    弹出文件选择框，让用户选择Excel文件
    title: 弹窗标题
    return: 选中的文件路径，取消则返回None
    """
    try:
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if not file_path:
            print("未选择文件，程序退出！")
            exit()
        return file_path
    except Exception as e:
        print(f"文件选择出错：{e}")
        exit()

def read_basic_info():
    """
    纯读取：基准电压、日期、节点数据（无任何计算）
    sheet1: 日期（202511091200）
    sheet2: slack - 基准电压
    sheet3: 节点数据（节点号、有功值、无功值）
    return: UB(基准电压), date_str(解析后的日期), Bus(节点数据数组)
    """
    file_path = select_excel_file("选择【基准电压+节点数据】Excel文件")
    
    # 读取日期sheet（第一个sheet）
    df_date = pd.read_excel(file_path, sheet_name=0, header=None)
    date_raw = str(df_date.iloc[0, 0]).strip()  # 取第一个单元格的日期字符串
    # 解析日期（202511091200 → 2025-11-09 12:00）
    try:
        date_obj = datetime.strptime(date_raw, "%Y%m%d%H%M")
        date_str = date_obj.strftime("%Y年%m月%d日 %H:%M")
    except:
        date_str = date_raw  # 解析失败则保留原始字符串
    print(f"当前工况日期：{date_str}")
    
    # 读取基准电压sheet（第二个sheet）
    df_ub = pd.read_excel(file_path, sheet_name=1)
    UB = float(df_ub.iloc[0, 0])  # 第二行第一列是基准电压值
    print(f"读取到基准电压：{UB} kV")
    
    # 读取节点数据sheet（第三个sheet）
    df_bus = pd.read_excel(file_path, sheet_name=2)
    # 清理列名（去除空格）
    df_bus.columns = [col.strip() for col in df_bus.columns]
    bus_data = df_bus[['节点号', '有功值', '无功值']].values  # 提取核心列
    Bus = bus_data.flatten()  # 展平为一维数组，兼容原有逻辑
    print(f"读取到节点数据：共{len(bus_data)}个节点")
    
    return UB, date_str, Bus

def read_branch_data():
    """
    纯读取：线路参数（无任何计算）
    sheet1: 线路数据（线路号、首节点、末节点、电阻、电抗）
    return: Branch(线路参数数组)
    """
    file_path = select_excel_file("选择【线路参数】Excel文件")
    df_branch = pd.read_excel(file_path, sheet_name=0)
    df_branch.columns = [col.strip() for col in df_branch.columns]
    branch_data = df_branch[['线路号', '首节点', '末节点', '电阻', '电抗']].values
    Branch = branch_data.flatten()  # 展平为一维数组
    print(f"读取到线路数据：共{len(branch_data)}条线路")
    return Branch

def read_pv_raw_data():
    """
    纯读取：光伏可调节点原始数据（仅读取，无任何计算/处理）
    sheet1: 光伏节点数据（节点号、容量、调度命名）
    return: pv_raw_data (列表，每个元素是(光伏节点号, 容量, 调度命名))
    """
    file_path = select_excel_file("选择【光伏可调节点】Excel文件")
    df_pv = pd.read_excel(file_path, sheet_name=0)
    
    # 清理列名（去除空格、特殊字符、单位）
    df_pv.columns = [
        col.strip().replace('(', '').replace(')', '').replace('（', '').replace('）', '')
        .replace('MVA', '').replace('kW', '').replace('MW', '').replace(' ', '')
        for col in df_pv.columns
    ]
    
    # 打印Excel中的实际列名，方便排查
    print("\n=== 光伏Excel文件中的实际列名 ===")
    for idx, col in enumerate(df_pv.columns):
        print(f"列{idx+1}：{col}")
    
    # 适配多种容量列名
    capacity_col = None
    possible_capacity_cols = ['容量', '装机容量', '光伏容量', '额定容量', '总容量']
    for col in possible_capacity_cols:
        if col in df_pv.columns:
            capacity_col = col
            break
    if not capacity_col:
        raise ValueError(f"未找到容量相关列名！请检查Excel列名，支持的列名：{possible_capacity_cols}")
    
    # 适配多种节点号/命名列名
    node_num_col = '节点号' if '节点号' in df_pv.columns else (
        '光伏节点号' if '光伏节点号' in df_pv.columns else df_pv.columns[0]
    )
    name_col = '调度命名' if '调度命名' in df_pv.columns else (
        '节点名称' if '节点名称' in df_pv.columns else df_pv.columns[2]
    )
    
    # 仅读取原始数据，不做任何计算
    pv_raw_data = []
    for idx, row in df_pv.iterrows():
        pv_node_num = int(row[node_num_col])    # 光伏节点号（自然序号）
        pv_capacity = float(row[capacity_col])  # 光伏容量(MVA)
        pv_name = str(row[name_col]).strip()    # 调度命名
        pv_raw_data.append((pv_node_num, pv_capacity, pv_name))
    
    print(f"\n读取到光伏原始数据：共{len(pv_raw_data)}个光伏节点")
    for pv_node_num, pv_capacity, pv_name in pv_raw_data:
        print(f"  节点{pv_node_num}({pv_name})：容量={pv_capacity} MVA")
    
    return pv_raw_data

def read_voltage_limits():
    """
    纯读取：电压上下限（无任何计算）
    sheet1: 第二行开始是电压下限、上限（两列）
    return: v_min, v_max
    """
    file_path = select_excel_file("选择【电压上下限】Excel文件")
    df_v = pd.read_excel(file_path, sheet_name=0, header=None)
    v_min = float(df_v.iloc[1, 0])  # 第二行第一列：电压下限
    v_max = float(df_v.iloc[1, 1])  # 第二行第二列：电压上限
    print(f"\n读取到电压约束：下限={v_min} kV，上限={v_max} kV")
    return v_min, v_max

# -------------------------- 光伏数据处理函数（主程序调用） --------------------------
def process_pv_data_to_tunable_q(pv_raw_data, Bus_mat):
    """
    处理光伏原始数据，生成tunable_q_nodes（主程序中调用）
    pv_raw_data: read_pv_raw_data返回的原始数据
    Bus_mat: 节点数据二维矩阵
    return: tunable_q_nodes(可调无功节点配置列表)
    """
    tunable_q_nodes = []
    for pv_node_num, pv_capacity, pv_name in pv_raw_data:
        # 1. 光伏节点号减1 → 数组索引
        node_idx = pv_node_num - 1
        # 2. 从Bus_mat获取该节点的有功值（第二列，索引1）
        p_node = Bus_mat[node_idx, 1]
        # 3. 计算无功上下限：q = sqrt(容量² - 有功²)
        q_limit = np.sqrt(pv_capacity**2 - p_node**2)
        q_min = -q_limit
        q_max = q_limit
        
        # 4. 构建可调节点配置
        tunable_q_nodes.append((node_idx, q_min, q_max, pv_name))
        print(f"\n光伏节点{pv_node_num}({pv_name}) 处理结果：")
        print(f"  有功={p_node} MW，无功上下限=[{q_min:.4f}, {q_max:.4f}] Mvar")
    
    return tunable_q_nodes

# -------------------------- 核心参数初始化（读取+处理分离） --------------------------
# 1. 纯读取：基准电压、节点数据、日期
UB, date_str, Bus = read_basic_info()

# 2. 纯读取：线路参数
Branch = read_branch_data()

# 3. 重塑节点数据为二维矩阵
Bus_mat = Bus.reshape(-1, 3)

# 4. 纯读取：光伏原始数据 → 处理：生成tunable_q_nodes（分离核心）
pv_raw_data = read_pv_raw_data()  # 仅读取
tunable_q_nodes = process_pv_data_to_tunable_q(pv_raw_data, Bus_mat)  # 单独处理

# 5. 纯读取：电压上下限
v_min, v_max = read_voltage_limits()

# 通用配置（其余保留）
pr = 1e-6  # 潮流收敛精度
SB = 10    # 基准功率 单位 MVA

# 设置中文显示（适配更多系统）
plt.rcParams["font.family"] = ["Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# -------------------------- 潮流计算函数（修改：增加bus_override参数） --------------------------
def power_flow(tunable_q_values, bus_override=None):
    """
    潮流计算函数（通用版）
    tunable_q_values: 可调节点的无功值列表，顺序与tunable_q_nodes一致
    bus_override: 可选，覆盖的节点数据矩阵（二维），用于强制修改所有节点参数（如无功全0）
    """
    # 复制原始数据（支持自定义节点数据覆盖）
    if bus_override is not None:
        # 使用传入的自定义节点数据（已处理好所有参数）
        Bus_copy = bus_override.flatten()
    else:
        Bus_copy = copy.deepcopy(Bus)
    Branch_copy = copy.deepcopy(Branch)
    
    # 仅当没有自定义节点数据时，才更新可调节点的无功值
    if bus_override is None:
        # 更新可调节点的无功值（核心：根据配置动态更新）
        for i, (node_idx, _, _, _) in enumerate(tunable_q_nodes):
            Bus_copy = Bus_copy.reshape(-1, 3)  # 临时重塑
            Bus_copy[node_idx, 2] = tunable_q_values[i]  # 第2列是无功
            Bus_copy = Bus_copy.flatten()  # 还原为一维
    
    # 标幺化处理
    Bus_copy = Bus_copy.reshape(-1, 3)
    Bus_copy[:, 1] = Bus_copy[:, 1] / SB  # 有功标幺值
    Bus_copy[:, 2] = Bus_copy[:, 2] / SB  # 无功标幺值
    Branch_copy = Branch_copy.reshape(-1, 5)
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
    TempBranch = Branch_copy.copy()
    s1 = np.zeros((0, 5))
    while TempBranch.size > 0:
        s = TempBranch.shape[0] - 1
        s2 = np.zeros((0, 5))
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

# -------------------------- 粒子群优化算法（通用版） --------------------------
def pso_optimization():
    # 算法参数
    num_particles = 50  # 粒子数量
    max_iter = 50       # 最大迭代次数
    w = 0.8             # 初始惯性权重
    c1 = 1.5            # 认知系数
    c2 = 1.5            # 社会系数
    
    # 从配置中提取可调节点参数（动态适应维度）
    dim = len(tunable_q_nodes)  # 决策变量维度（可调节点数量）
    q_mins = np.array([node[1] for node in tunable_q_nodes])  # 各节点最小值
    q_maxs = np.array([node[2] for node in tunable_q_nodes])  # 各节点最大值
    
    # 初始化粒子群（动态维度）
    particles = np.random.rand(num_particles, dim)  # 粒子位置（num_particles × dim）
    for i in range(dim):
        # 映射到各节点的无功范围
        particles[:, i] = particles[:, i] * (q_maxs[i] - q_mins[i]) + q_mins[i]
    
    velocities = np.zeros((num_particles, dim))  # 粒子速度（动态维度）
    
    # 初始化个体最优和全局最优
    pbest = np.copy(particles)
    pbest_fitness = np.ones(num_particles) * np.inf
    
    # 计算初始适应度
    for i in range(num_particles):
        loss_rate, voltages, _ = power_flow(particles[i])  # 传入当前粒子的所有无功值
        if loss_rate is None:
            pbest_fitness[i] = np.inf
            continue
        
        # 电压越限惩罚
        voltage_violation = np.sum(np.maximum(v_min - voltages, 0) + np.maximum(voltages - v_max, 0))
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
            # 速度更新（动态维度）
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
            # 位置限制（各节点独立约束）
            for j in range(dim):
                particles[i, j] = np.clip(particles[i, j], q_mins[j], q_maxs[j])
            
            # 计算当前适应度
            loss_rate, voltages, _ = power_flow(particles[i])
            if loss_rate is None:
                current_fitness = np.inf
            else:
                voltage_violation = np.sum(np.maximum(v_min - voltages, 0) + np.maximum(voltages - v_max, 0))
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
            print(f"迭代 {iter + 1}/{max_iter}, 最优适应度: {fitness_history[-1]:.4f}%")
    
    return gbest, gbest_fitness, fitness_history

# -------------------------- 主函数（新增理论极限网损计算） --------------------------
def main():
    print("===== 网损优化计算程序 =====")
    # 提取可调节点的初始无功值、名称、节点号（用于结果展示）
    initial_q_values = []
    node_names = []
    node_numbers = []  # 新增：存储原始节点号（不是索引）
    for node_idx, _, _, name in tunable_q_nodes:
        Bus_2d = Bus.reshape(-1, 3)
        initial_q = Bus_2d[node_idx, 2]  # 原始无功值
        initial_q_values.append(initial_q)
        node_names.append(name)
        node_number = node_idx + 1  # 还原为原始节点号（索引+1）
        node_numbers.append(node_number)  # 保存原始节点号
    dim = len(tunable_q_nodes)
    
    # 打印初始状态
    print("\n===== 初始状态 =====")
    for i in range(dim):
        print(f"{node_numbers[i]}（{node_names[i]}） 初始无功 = {initial_q_values[i]:.4f} Mvar")
    
    # 初始潮流计算
    initial_loss_rate, initial_voltages, _ = power_flow(initial_q_values)
    if initial_loss_rate is None:
        print("初始潮流计算不收敛！")
        return
    print(f"初始网损率: {initial_loss_rate:.4f}%")
    
    # -------------------------- 新增：计算理论极限网损（所有节点无功=0） --------------------------
    print("\n===== 计算理论极限网损（所有节点无功=0） =====")
    # 创建所有无功为0的节点矩阵
    Bus_limit_mat = copy.deepcopy(Bus_mat)
    Bus_limit_mat[:, 2] = 0.0  # 所有节点的无功列（第三列）置0
    # 调用潮流计算，传入自定义节点数据（无功全0）
    limit_loss_rate, limit_voltages, _ = power_flow(initial_q_values, bus_override=Bus_limit_mat)
    if limit_loss_rate is None:
        print("理论极限网损计算不收敛！")
        return
    print(f"理论极限网损率（所有节点无功=0）: {limit_loss_rate:.4f}%")
    
    # 粒子群优化
    print("\n===== 开始粒子群优化 =====")
    optimal_params, optimal_loss, fitness_history = pso_optimization()
    
    # 优化后结果
    final_loss_rate, final_voltages, _ = power_flow(optimal_params)
    if final_loss_rate is None:
        print("优化后潮流计算不收敛！")
        return
    
    # 输出优化结果
    print("\n===== 优化结果 ======")
    print(f"工况日期：{date_str}")
    print("优化前无功：")
    for i in range(dim):
        # 格式：节点号（光伏名称）: 无功值 Mvar
        print(f"{node_numbers[i]}（{node_names[i]}）: {initial_q_values[i]:.4f} Mvar")
    
    print("\n优化后无功：")
    for i in range(dim):
        # 格式：节点号（光伏名称）: 无功值 Mvar
        print(f"{node_numbers[i]}（{node_names[i]}）: {optimal_params[i]:.4f} Mvar")
    
    # 新增：输出无功调节量（优化后 - 优化前）
    print("\n无功调节量（优化后 - 优化前）：")
    for i in range(dim):
        adjust_amount = optimal_params[i] - initial_q_values[i]
        print(f"{node_numbers[i]}（{node_names[i]}）: {adjust_amount:.4f} Mvar")
    
    # -------------------------- 扩展：输出理论极限网损对比 --------------------------
    print(f"\n=== 网损率对比 ===")
    print(f"优化前网损率: {initial_loss_rate:.4f}%")
    print(f"优化后网损率: {final_loss_rate:.4f}%")
    print(f"理论极限网损率（所有节点无功=0）: {limit_loss_rate:.4f}%")
    print(f"优化后相比初始降低: {initial_loss_rate - final_loss_rate:.4f}%")
    print(f"优化后相比初始降低百分比: {((initial_loss_rate - final_loss_rate)/initial_loss_rate)*100:.2f}%")
   # print(f"理论极限相比初始降低: {initial_loss_rate - limit_loss_rate:.4f}%")
  # print(f"理论极限相比初始降低百分比: {((initial_loss_rate - limit_loss_rate)/initial_loss_rate)*100:.2f}%")
    
    # 可视化1：优化过程（添加日期标题）
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history)+1), fitness_history, 'b-')
    plt.xlabel('迭代次数')
    plt.ylabel('最优网损率 (%)')
    plt.title(f'粒子群优化过程 - {date_str}')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    
    # 可视化2：网损率对比（新增理论极限维度）
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ['优化前', '优化后', '理论极限（无功全0）'], 
        [initial_loss_rate, final_loss_rate, limit_loss_rate],
        color=['red', 'green', 'blue']
    )
    plt.ylabel('网损率 (%)')
    plt.title(f'优化前后及理论极限网损率对比 - {date_str}')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}%', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # 可视化3：电压分布对比
    nodes = np.arange(1, len(initial_voltages)+1)
    plt.figure(figsize=(12, 6))
    plt.plot(nodes, initial_voltages, 'r-', marker='o', label='优化前')
    plt.plot(nodes, final_voltages, 'g-', marker='s', label='优化后')
    plt.plot(nodes, limit_voltages, 'b-', marker='^', label='理论极限（无功全0）')  # 新增理论极限电压
    plt.axhline(y=v_max, color='k', linestyle='--', label=f'电压上限 ({v_max} kV)')
    plt.axhline(y=v_min, color='k', linestyle='-.', label=f'电压下限 ({v_min} kV)')
    plt.xlabel('节点号')
    plt.ylabel('电压 (kV)')
    plt.title(f'优化前后及理论极限节点电压分布对比 - {date_str}')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
    
    # 可视化4：可调节点无功对比
    plt.figure(figsize=(8, 6))
    width = 0.35
    x = np.arange(dim)
    plt.bar(x - width/2, initial_q_values, width, label='优化前')
    plt.bar(x + width/2, optimal_params, width, label='优化后')
    plt.xticks(x, [f"{node_numbers[i]}（{node_names[i]}）" for i in range(dim)], rotation=45, ha='right')
    plt.ylabel('无功 (Mvar)')
    plt.title(f'可调节光伏节点无功对比 - {date_str}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    main()