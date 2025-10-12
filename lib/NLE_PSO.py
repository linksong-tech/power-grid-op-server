import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy

# 网损经济


# 设置中文显示（适配更多系统）
plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]  # "Heiti TC"是 macOS 自带中文字体，sans-serif是后备
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# -------------------------- 通用配置参数 --------------------------
# 基准参数
SB = 10  # 基准功率 单位 MVA
UB = 10.38  # 基准电压 单位 kV
pr = 1e-6  # 潮流收敛精度

# 可调无功节点配置（核心修改：通过此列表指定可调节点）
# 格式：[(节点索引, 无功最小值(Mvar), 无功最大值(Mvar), 节点名称), ...]
# 节点索引：Bus数组中的行索引（从0开始）
tunable_q_nodes = [
    (32, -0.3, 0.3, "节点33"),  # 节点33（原代码索引32）
    (33, -0.5, 0.5, "节点34")   # 节点34（原代码索引33）
]
# ------------------------------------------------------------------

# 节点数据 [节点号, 负荷有功(MW), 负荷无功(Mvar)]
Bus = np.array([
    1, 0,  0,          
    2, 0,  0,         
    3, 0,  0,         
    4, 0,  0,          
    5, 0,  0,          
    6, 0,  0,          
    7, 0,  0,         
    8, 0,  0,  
    9, 0,  0,          
    10,0,  0, 
    11,0.325071,-0.023704,
    12,0,0,
    13,-0.02834,0,
    14,0.20565,0.00301,
    15,0,0,
    16,-0.067145,0,
    17,0.095275,0.00102,
    18,0,0,
    19,-0.09425,0,
    20,0.24197,0.02307,
    21,0,0,
    22,-0.052715,0,
    23,0.214965,0.05159,
    24,0,0,
    25,-0.1508,0,
    26,0.231172,0.007432,
    27,0.121383,0.02996,
    28,0,0,
    29,-0.47385,0,
    30,0.648046,0.163962,
    31,0,0,
    32,0.01908,0.00991,
    33,-0.6704,0.0386 ,     #可调光伏
    34,-1.205,0.414,	       #可调光伏
])

# 支路数据 [支路号, 首节点, 尾节点, 电阻(Ω), 电抗(Ω)]
Branch = np.array([
    1,1,2,0.3436,0.8136,
    2,2,3,0.0341,0.0807,
    3,3,4,0.0369,0.0874,
    4,4,5,0.0426,0.1009,
    5,5,6,0.0852,0.2017,
    6,6,7,0.4885,1.1565,
    7,7,8,0.7128,1.6877,
    8,8,9,0.3280,0.7766,
    9,9,10,0.3124,0.7396,
    10,10,11,0.4686,1.1095,
    11,2,12,0.2456,0.1778,
    12,12,13,0.0238,0.0172,
    13,12,14,0.0238,0.0172,
    14,3,15,0.0475,0.0344,
    15,15,16,0.0238,0.0172,
    16,15,17,0.0238,0.0172,
    17,4,18,0.4275,0.3096,
    18,18,19,0.0238,0.0172,
    19,18,20,0.0238,0.0172,
    20,5,21,0.2233,0.1617,
    21,21,22,0.0238,0.0172,
    22,21,23,0.0238,0.0172,
    23,6,24,0.4275,0.3096,
    24,24,25,0.02375,0.0172,
    25,24,26,0.02375,0.0172,
    26,7,27,0.35625,0.258,
    27,8,28,0.4061,1.0168,
    28,28,29,0.02375,0.0172,
    29,28,30,0.02375,0.0172,
    30,9,31,0.475,0.344,
    31,31,32,0.02375,0.0172,
    32,31,33,0.02375,0.0172,
    33,10,34,0.16625,0.1204
])

# 数据重塑（二维化）
Buscols = 3
Bus = Bus.reshape(-1, Buscols)
Branchcols = 5
Branch = Branch.reshape(-1, Branchcols)

# -------------------------- 潮流计算函数 --------------------------
def power_flow(tunable_q_values):
    """
    潮流计算函数（通用版）
    tunable_q_values: 可调节点的无功值列表，顺序与tunable_q_nodes一致
    """
    # 复制原始数据
    Bus_copy = copy.deepcopy(Bus)
    Branch_copy = copy.deepcopy(Branch)
    
    # 更新可调节点的无功值（核心：根据配置动态更新）
    for i, (node_idx, _, _, _) in enumerate(tunable_q_nodes):
        Bus_copy[node_idx, 2] = tunable_q_values[i]  # 第2列是无功
    
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

# -------------------------- 粒子群优化算法（通用版） --------------------------
def pso_optimization():
    # 算法参数
    num_particles = 30  # 粒子数量
    max_iter = 50       # 最大迭代次数
    w = 0.8             # 初始惯性权重
    c1 = 1.5            # 认知系数
    c2 = 1.5            # 社会系数
    
    # 从配置中提取可调节点参数（动态适应维度）
    dim = len(tunable_q_nodes)  # 决策变量维度（可调节点数量）
    q_mins = np.array([node[1] for node in tunable_q_nodes])  # 各节点最小值
    q_maxs = np.array([node[2] for node in tunable_q_nodes])  # 各节点最大值
    
    # 电压约束
    v_min = 0.95 * UB
    v_max = 1.05 * UB
    
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
            print(f"迭代 {iter + 1}/{max_iter}, 最优网损率: {fitness_history[-1]:.4f}%")
    
    return gbest, gbest_fitness, fitness_history

# -------------------------- 主函数 --------------------------
def main():
    # 提取可调节点的初始无功值和名称（用于结果展示）
    initial_q_values = [Bus[node[0], 2] for node in tunable_q_nodes]
    node_names = [node[3] for node in tunable_q_nodes]
    dim = len(tunable_q_nodes)
    
    # 打印初始状态
    print("初始状态：")
    for i in range(dim):
        print(f"{node_names[i]} 初始无功 = {initial_q_values[i]:.4f} Mvar")
    
    # 初始潮流计算
    initial_loss_rate, initial_voltages, _ = power_flow(initial_q_values)
    if initial_loss_rate is None:
        print("初始潮流计算不收敛！")
        return
    print(f"初始网损率: {initial_loss_rate:.4f}%")
    
    # 粒子群优化
    print("\n开始粒子群优化...")
    optimal_params, optimal_loss, fitness_history = pso_optimization()
    
    # 优化后结果
    final_loss_rate, final_voltages, _ = power_flow(optimal_params)
    if final_loss_rate is None:
        print("优化后潮流计算不收敛！")
        return
    
    # 输出优化结果
    print("\n===== 优化结果 =====")
    print("优化前无功：")
    for i in range(dim):
        print(f"{node_names[i]}: {initial_q_values[i]:.4f} Mvar")
    print("\n优化后无功：")
    for i in range(dim):
        print(f"{node_names[i]}: {optimal_params[i]:.4f} Mvar")
    
    print(f"\n优化前网损率: {initial_loss_rate:.4f}%")
    print(f"优化后网损率: {final_loss_rate:.4f}%")
    print(f"网损率降低: {initial_loss_rate - final_loss_rate:.4f}%")
    print(f"网损率降低百分比: {((initial_loss_rate - final_loss_rate)/initial_loss_rate)*100:.2f}%")
    
    # 可视化1：优化过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history)+1), fitness_history, 'b-')
    plt.xlabel('迭代次数')
    plt.ylabel('最优网损率 (%)')
    plt.title('粒子群优化过程')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    
    # 可视化2：网损率对比
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['优化前', '优化后'], [initial_loss_rate, final_loss_rate], color=['red', 'green'])
    plt.ylabel('网损率 (%)')
    plt.title('优化前后网损率对比')
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
    plt.axhline(y=0.95*UB, color='k', linestyle='--', label='电压下限 (0.95pu)')
    plt.axhline(y=1.05*UB, color='k', linestyle='-.', label='电压上限 (1.05pu)')
    plt.xlabel('节点号')
    plt.ylabel('电压 (kV)')
    plt.title('优化前后节点电压分布对比')
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
    plt.xticks(x, node_names)
    plt.ylabel('无功 (Mvar)')
    plt.title('可调节点无功对比')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()