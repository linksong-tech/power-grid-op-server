import numpy as np
import json

def power_flow_calculation(Bus, Branch, SB=10, UB=10.38, pr=1e-6):
    """
    前推回代法潮流计算通用函数
    
    参数:
    Bus: 节点数据 numpy数组，格式[节点号, 负荷有功(MW), 负荷无功(Mvar)]
    Branch: 支路数据 numpy数组，格式[支路号, 首节点, 尾节点, 电阻(Ω), 电抗(Ω)]
    SB: 基准功率(MVA)，默认10
    UB: 基准电压(kV)，默认10.38
    pr: 收敛精度，默认1e-6
    
    返回:
    json格式的计算结果
    """
    try:
        # 数据重塑
        Buscols = 3
        Bus = Bus.reshape(-1, Buscols)
        Branchcols = 5
        Branch = Branch.reshape(-1, Branchcols)
        
        # 获取节点和支路数量（自适应）
        busnum = Bus.shape[0]
        branchnum = Branch.shape[0]
        
        # 规范化支路编号，确保为 1..branchnum，避免由外部不连续/过大编号导致的越界
        # 后续所有以支路号为索引的数组（如 Ploss、Qloss、P、Q、I）均与此顺序对齐
        normalized_branch_ids = np.arange(1, branchnum + 1)
        Branch[:, 0] = normalized_branch_ids
        
        # 标幺化处理
        Bus_pu = np.copy(Bus)
        Bus_pu[:, 1] = Bus_pu[:, 1] / SB  # 负荷有功标幺值
        Bus_pu[:, 2] = Bus_pu[:, 2] / SB  # 负荷无功标幺值
        
        Branch_pu = np.copy(Branch)
        Branch_pu[:, 3] = Branch_pu[:, 3] * SB / (UB ** 2)  # 电阻标幺值
        Branch_pu[:, 4] = Branch_pu[:, 4] * SB / (UB ** 2)  # 电抗标幺值
        
        # 标识节点类型
        node_types = []
        for i in range(busnum):
            node_id = Bus_pu[i, 0]
            p = Bus_pu[i, 1]  # 标幺值有功
            if node_id == 1:  # 平衡节点（发电机节点）
                node_types.append("平衡节点")
            elif p < 0:  # 光伏节点（有功注入，标幺值为负）
                node_types.append("光伏节点")
            elif p > 0:  # 负荷节点（有功消耗，标幺值为正）
                node_types.append("负荷节点")
            else:  # 普通节点（无有功注入/消耗）
                node_types.append("普通节点")
        
        # 初始化节点电压和相角
        Vbus = np.ones(busnum)  # 电压幅值初始化为1
        Vbus[0] = 1.0  # 平衡节点电压
        cita = np.zeros(busnum)  # 电压相角
        
        # 初始化变量
        k = 0  # 迭代次数
        Ploss = np.zeros(branchnum)  # 支路有功损耗
        Qloss = np.zeros(branchnum)  # 支路无功损耗
        P = np.zeros(branchnum)  # 支路有功
        Q = np.zeros(branchnum)  # 支路无功
        I = np.zeros(branchnum)  # 支路电流
        F = 0  # 收敛标志
        
        # 支路排序（从末端到首端）
        TempBranch = Branch_pu.copy()
        s1 = np.zeros((0, Branchcols))  # 存储排序后的支路
        
        while TempBranch.size > 0:
            s = TempBranch.shape[0] - 1
            s2 = np.zeros((0, Branchcols))
            
            while s >= 0:
                # 检查是否为末端节点（没有其他支路以该节点为首节点）
                i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
                if i.size == 0:  # 末端节点
                    s1 = np.vstack([s1, TempBranch[s, :]])
                else:  # 非末端节点
                    if s2.size > 0:
                        s2 = np.vstack([s2, TempBranch[s, :]])
                    else:
                        s2 = TempBranch[s, :].reshape(1, -1)
                s -= 1
            
            TempBranch = s2.copy()
        
        # 潮流计算迭代
        max_iter = 100
        while k < max_iter and F == 0:
            Pij1 = np.zeros(busnum)  # 节点注入有功
            Qij1 = np.zeros(busnum)  # 节点注入无功
            
            # 前推过程：计算支路功率
            for s in range(branchnum):
                ii = int(s1[s, 1] - 1)  # 首节点索引
                jj = int(s1[s, 2] - 1)  # 尾节点索引
                Pload = Bus_pu[jj, 1]      # 节点有功负荷
                Qload = Bus_pu[jj, 2]      # 节点无功负荷
                R = s1[s, 3]            # 支路电阻
                X = s1[s, 4]            # 支路电抗
                VV = Vbus[jj]           # 节点电压
                
                # 后续支路功率
                Pij0 = Pij1[jj]
                Qij0 = Qij1[jj]
      
                # 计算电流平方和损耗
                II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
                Ploss[int(s1[s, 0]) - 1] = II * R
                Qloss[int(s1[s, 0]) - 1] = II * X
                
                # 计算支路功率
                P[int(s1[s, 0]) - 1] = Pload + Ploss[int(s1[s, 0]) - 1] + Pij0
                Q[int(s1[s, 0]) - 1] = Qload + Qloss[int(s1[s, 0]) - 1] + Qij0
                
                # 累加到首节点
                Pij1[ii] += P[int(s1[s, 0]) - 1]
                Qij1[ii] += Q[int(s1[s, 0]) - 1]
            
            # 回推过程：计算节点电压
            for s in range(branchnum-1, -1, -1):
                ii = int(s1[s, 2] - 1)  # 尾节点索引
                kk = int(s1[s, 1] - 1)  # 首节点索引
                R = s1[s, 3]            # 支路电阻
                X = s1[s, 4]            # 支路电抗
                
                # 计算电压分量
                V_real = Vbus[kk] - (P[int(s1[s, 0]) - 1]*R + Q[int(s1[s, 0]) - 1]*X) / Vbus[kk]
                V_imag = (P[int(s1[s, 0]) - 1]*X - Q[int(s1[s, 0]) - 1]*R) / Vbus[kk]
                
                # 更新节点电压幅值和相角
                Vbus[ii] = np.sqrt(V_real**2 + V_imag**2)
                cita[ii] = cita[kk] - np.arctan2(V_imag, V_real)
            
            # 再次前推计算（用于收敛判断）
            Pij2 = np.zeros(busnum)
            Qij2 = np.zeros(busnum)
            
            for s in range(branchnum):
                ii = int(s1[s, 1] - 1)  # 首节点索引
                jj = int(s1[s, 2] - 1)  # 尾节点索引
                Pload = Bus_pu[jj, 1]      # 节点有功负荷
                Qload = Bus_pu[jj, 2]      # 节点无功负荷
                R = s1[s, 3]            # 支路电阻
                X = s1[s, 4]            # 支路电抗
                VV = Vbus[jj]           # 节点电压
                
                # 后续支路功率
                Pij0 = Pij2[jj]
                Qij0 = Qij2[jj]
                
                # 计算电流和损耗
                II = ((Pload + Pij0)**2 + (Qload + Qij0)** 2) / (VV**2)
                I[int(s1[s, 0]) - 1] = np.sqrt(II) * 1000  # 转换为安培
                
                # 计算支路功率
                P_val = Pload + II * R + Pij0
                Q_val = Qload + II * X + Qij0
                P[int(s1[s, 0]) - 1] = P_val
                Q[int(s1[s, 0]) - 1] = Q_val
                
                # 累加到首节点
                Pij2[ii] += P_val
                Qij2[ii] += Q_val
            
            # 检查收敛条件
            ddp = np.max(np.abs(Pij1 - Pij2))
            ddq = np.max(np.abs(Qij1 - Qij2))

            
            if ddp < pr and ddq < pr:
                F = 1
            
            k += 1
        
        # 检查是否收敛
        if k >= max_iter and F == 0:
            return json.dumps({
                "status": "error",
                "message": "潮流计算不收敛"
            }, ensure_ascii=False)
        
        # 计算总损耗
        P1 = np.sum(Ploss)
        Q1 = np.sum(Qloss)
        
        # 提取节点号和计算结果（转换为有名值）
        node_numbers = Bus[:, 0].astype(int)
        Vbus_kv = Vbus * UB  # 电压幅值（kV，有名值）
        angles_deg = np.degrees(cita)  # 相角（度）
        P_mw = P * SB  # 支路有功（MW，有名值）
        Q_mvar = Q * SB  # 支路无功（MVar，有名值）
        load_P_mw = Bus[:, 1]  # 节点负荷有功（MW，有名值）
        load_Q_mvar = Bus[:, 2]  # 节点负荷无功（MVar，有名值）
        
        # 网损率计算
        balance_node_output = Pij2[0] * SB  # 平衡节点输出功率（MW）
        pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
        pv_total_injection = sum(-Bus_pu[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * SB  # 光伏总注入（MW）
        total_input_power = balance_node_output + pv_total_injection
        
        load_nodes_mask = [typ == "负荷节点" for typ in node_types]
        total_output_power = sum(Bus_pu[i, 1] for i in range(busnum) if load_nodes_mask[i]) * SB  # 负荷总消耗（MW）
        
        loss_rate = (total_input_power - total_output_power) / total_input_power * 100 if total_input_power != 0 else 0.0
        
        # 整理节点结果
        node_results = []
        for i in range(busnum):
            node_results.append({
                "node_id": int(node_numbers[i]),
                "node_type": node_types[i],
                "voltage_kv": float(Vbus_kv[i]),
                "angle_deg": float(angles_deg[i]),
                "load_p_mw": float(load_P_mw[i]),
                "load_q_mvar": float(load_Q_mvar[i])
            })
        
        # 整理支路结果
        branch_results = []
        for i in range(branchnum):
            from_node = int(Branch[i, 1])
            to_node = int(Branch[i, 2])
            branch_results.append({
                "branch_id": int(Branch[i, 0]),
                "from_node": from_node,
                "to_node": to_node,
                "active_power_mw": float(P_mw[i]),
                "reactive_power_mvar": float(Q_mvar[i]),
                "active_loss_mw": float(Ploss[i] * SB),
                "reactive_loss_mvar": float(Qloss[i] * SB)
            })
        
        # 整理汇总结果
        summary = {
            "total_active_loss_mw": float(P1 * SB),
            "total_reactive_loss_mvar": float(Q1 * SB),
            "balance_node_output_mw": float(balance_node_output),
            "pv_total_injection_mw": float(pv_total_injection),
            "total_input_power_mw": float(total_input_power),
            "total_output_power_mw": float(total_output_power),
            "loss_rate_percent": float(loss_rate),
            "iteration_count": k,
            "converged": bool(F)
        }
        
        # 构建返回结果
        result = {
            "status": "success",
            "parameters": {
                "base_power_mva": SB,
                "base_voltage_kv": UB,
                "precision": pr
            },
            "node_results": node_results,
            "branch_results": branch_results,
            "summary": summary
        }
        
        # 转换为JSON并返回
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)


# 示例用法（实际Web应用中可移除）
if __name__ == "__main__":
    # 示例数据
    SB = 10  # 基准功率 单位 MVA
    UB = 10.38  # 基准电压 单位 kV
    pr = 1e-6  # 收敛精度
    
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
        33,-0.6704,-0.1258 ,     #可调光伏
        34,-1.205,-0.1140,	       #可调光伏
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
    
    # 调用计算函数
    result = power_flow_calculation(Bus, Branch, SB, UB, pr)
    print(result)