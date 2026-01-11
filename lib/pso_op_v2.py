import copy
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    # When imported with lib/ on sys.path (e.g. routes add lib to sys.path).
    from voltage_bounds import compute_voltage_bounds_kv
except ModuleNotFoundError:  # pragma: no cover
    # When imported as a package module (e.g. import lib.pso_op_v2).
    from .voltage_bounds import compute_voltage_bounds_kv


def _as_2d_float_array(arr, cols: int, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != cols:
        raise ValueError(f"{name} must be a 2D array with shape (n, {cols}), got {a.shape}")
    return a


def _power_flow_forward_backward_sweep(
    *,
    Bus: np.ndarray,
    Branch: np.ndarray,
    tunable_q_nodes: Sequence[Sequence],
    tunable_q_values: Sequence[float],
    SB: float,
    UB: float,
    pr: float,
    bus_override: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    """
    Radial power flow (forward/backward sweep), consistent with the existing engineered PSO.

    Returns:
      (loss_rate_percent, vbus_kv, (balance_output_mw, pv_total_injection_mw, total_input_mw, total_output_mw))
      If not converged: (None, None, None)
    """
    Bus_copy = copy.deepcopy(Bus) if bus_override is None else _as_2d_float_array(bus_override, 3, "bus_override")
    Branch_copy = copy.deepcopy(Branch)

    # Update tunable Q (only if not overridden)
    if bus_override is None:
        for i, node in enumerate(tunable_q_nodes):
            node_idx = int(node[0])
            Bus_copy[node_idx, 2] = float(tunable_q_values[i])

    # Normalize branch IDs to 1..branchnum to avoid external indexing issues
    branchnum = Branch_copy.shape[0]
    Branch_copy[:, 0] = np.arange(1, branchnum + 1, dtype=float)

    # Per-unit conversion
    Bus_copy[:, 1] = Bus_copy[:, 1] / SB
    Bus_copy[:, 2] = Bus_copy[:, 2] / SB
    Branch_copy[:, 3] = Branch_copy[:, 3] * SB / (UB**2)
    Branch_copy[:, 4] = Branch_copy[:, 4] * SB / (UB**2)

    busnum = Bus_copy.shape[0]

    node_types: List[str] = []
    for i in range(busnum):
        node_id = Bus_copy[i, 0]
        p = Bus_copy[i, 1]
        if int(node_id) == 1:
            node_types.append("平衡节点")
        elif p < 0:
            node_types.append("光伏节点")
        elif p > 0:
            node_types.append("负荷节点")
        else:
            node_types.append("普通节点")

    Vbus = np.ones(busnum, dtype=float)
    Vbus[0] = 1.0
    cita = np.zeros(busnum, dtype=float)

    k = 0
    Ploss = np.zeros(branchnum, dtype=float)
    Qloss = np.zeros(branchnum, dtype=float)
    P = np.zeros(branchnum, dtype=float)
    Q = np.zeros(branchnum, dtype=float)
    converged = False

    # Branch sorting: from leaves to root (matches existing implementation)
    TempBranch = Branch_copy.copy()
    s1 = np.zeros((0, 5), dtype=float)
    while TempBranch.size > 0:
        s = TempBranch.shape[0] - 1
        s2 = np.zeros((0, 5), dtype=float)
        while s >= 0:
            i = np.where(TempBranch[:, 1] == TempBranch[s, 2])[0]
            if i.size == 0:
                s1 = np.vstack([s1, TempBranch[s, :]])
            else:
                s2 = np.vstack([s2, TempBranch[s, :]]) if s2.size > 0 else TempBranch[s, :].reshape(1, -1)
            s -= 1
        TempBranch = s2.copy()

    while k < 100 and not converged:
        Pij1 = np.zeros(busnum, dtype=float)
        Qij1 = np.zeros(busnum, dtype=float)

        # Forward sweep
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

            II = ((Pload + Pij0) ** 2 + (Qload + Qij0) ** 2) / (VV**2)
            branch_idx = int(s1[s, 0]) - 1
            Ploss[branch_idx] = II * R
            Qloss[branch_idx] = II * X

            P[branch_idx] = Pload + Ploss[branch_idx] + Pij0
            Q[branch_idx] = Qload + Qloss[branch_idx] + Qij0

            Pij1[ii] += P[branch_idx]
            Qij1[ii] += Q[branch_idx]

        # Backward sweep
        for s in range(branchnum - 1, -1, -1):
            ii = int(s1[s, 2] - 1)
            kk = int(s1[s, 1] - 1)
            R = s1[s, 3]
            X = s1[s, 4]
            branch_idx = int(s1[s, 0]) - 1

            V_real = Vbus[kk] - (P[branch_idx] * R + Q[branch_idx] * X) / Vbus[kk]
            V_imag = (P[branch_idx] * X - Q[branch_idx] * R) / Vbus[kk]
            Vbus[ii] = float(np.sqrt(V_real**2 + V_imag**2))
            cita[ii] = float(cita[kk] - np.arctan2(V_imag, V_real))

        # Convergence check
        Pij2 = np.zeros(busnum, dtype=float)
        Qij2 = np.zeros(busnum, dtype=float)
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

            II = ((Pload + Pij0) ** 2 + (Qload + Qij0) ** 2) / (VV**2)
            P_val = Pload + II * R + Pij0
            Q_val = Qload + II * X + Qij0

            Pij2[ii] += P_val
            Qij2[ii] += Q_val

        ddp = float(np.max(np.abs(Pij1 - Pij2)))
        ddq = float(np.max(np.abs(Qij1 - Qij2)))
        if ddp < pr and ddq < pr:
            converged = True
        k += 1

    if not converged:
        return None, None, None

    # Loss rate calculation
    balance_node_output = float(Pij2[0] * SB)
    pv_nodes_mask = [typ == "光伏节点" for typ in node_types]
    pv_total_injection = float(sum(-Bus_copy[i, 1] for i in range(busnum) if pv_nodes_mask[i]) * SB)
    total_input_power = float(balance_node_output + pv_total_injection)

    load_nodes_mask = [typ == "负荷节点" for typ in node_types]
    total_output_power = float(sum(Bus_copy[i, 1] for i in range(busnum) if load_nodes_mask[i]) * SB)

    loss_rate = (
        (total_input_power - total_output_power) / total_input_power * 100.0
        if total_input_power != 0
        else 0.0
    )
    Vbus_kv = Vbus * UB

    return loss_rate, Vbus_kv, (balance_node_output, pv_total_injection, total_input_power, total_output_power)


def pso_op(
    Bus,
    Branch,
    tunable_q_nodes,
    num_particles: int = 30,
    max_iter: int = 50,
    w: float = 0.8,
    c1: float = 1.5,
    c2: float = 1.5,
    v_min: float = 0.95,
    v_max: float = 1.05,
    SB: float = 10,
    UB: float = 10.38,
    pr: float = 1e-6,
):
    """
    基于粒子群算法(PSO)的电力系统无功优化（v2：基于 IntVQ_PSO 的工程化实现）

    入参尽量与旧版 `pso_op.py:pso_op()` 保持一致：
      - Bus: (n_bus, 3) [节点号, 负荷有功(MW), 负荷无功(Mvar)]
      - Branch: (n_branch, 5) [支路号, 首节点, 尾节点, 电阻(Ω), 电抗(Ω)]
      - tunable_q_nodes: [(节点索引, 无功最小值, 无功最大值, 节点名称), ...]

    返回：与旧版一致的结果 dict（新增字段不会破坏旧调用方）。
    """
    Bus_arr = _as_2d_float_array(Bus, 3, "Bus")
    Branch_arr = _as_2d_float_array(Branch, 5, "Branch")

    if not isinstance(tunable_q_nodes, (list, tuple)) or len(tunable_q_nodes) == 0:
        return {
            "error": "tunable_q_nodes 不能为空",
            "convergence": False,
            "optimal_params": [],
            "fitness_history": [],
        }

    dim = len(tunable_q_nodes)
    q_mins = np.array([float(node[1]) for node in tunable_q_nodes], dtype=float)
    q_maxs = np.array([float(node[2]) for node in tunable_q_nodes], dtype=float)

    if np.any(q_maxs < q_mins):
        return {
            "error": "tunable_q_nodes 中存在 q_max < q_min",
            "convergence": False,
            "optimal_params": [],
            "fitness_history": [],
        }

    v_bounds = compute_voltage_bounds_kv(v_min, v_max, UB)

    def eval_fitness(q_values: Sequence[float]) -> Tuple[float, Optional[float], Optional[np.ndarray], Optional[tuple]]:
        loss_rate, voltages_kv, power_info = _power_flow_forward_backward_sweep(
            Bus=Bus_arr,
            Branch=Branch_arr,
            tunable_q_nodes=tunable_q_nodes,
            tunable_q_values=q_values,
            SB=SB,
            UB=UB,
            pr=pr,
        )
        if loss_rate is None or voltages_kv is None:
            return float("inf"), None, None, None

        voltage_violation = float(
            np.sum(np.maximum(v_bounds.v_min_kv - voltages_kv, 0.0) + np.maximum(voltages_kv - v_bounds.v_max_kv, 0.0))
        )
        fitness = float(loss_rate + 100.0 * voltage_violation) if voltage_violation > 0 else float(loss_rate)
        return fitness, loss_rate, voltages_kv, power_info

    # Initial state (use Bus existing Q for tunable nodes)
    initial_q_values = [float(Bus_arr[int(node[0]), 2]) for node in tunable_q_nodes]
    _, initial_loss_rate, initial_voltages, _ = eval_fitness(initial_q_values)
    if initial_loss_rate is None or initial_voltages is None:
        return {
            "error": "初始潮流计算不收敛",
            "convergence": False,
            "optimal_params": [],
            "fitness_history": [],
            "initial_q_values": initial_q_values,
            "node_names": [node[3] for node in tunable_q_nodes],
        }

    # Optional: theoretical limit (all buses Q=0), matches IntVQ_PSO analysis
    limit_bus = copy.deepcopy(Bus_arr)
    limit_bus[:, 2] = 0.0
    limit_loss_rate, limit_voltages, _ = _power_flow_forward_backward_sweep(
        Bus=Bus_arr,
        Branch=Branch_arr,
        tunable_q_nodes=tunable_q_nodes,
        tunable_q_values=initial_q_values,
        SB=SB,
        UB=UB,
        pr=pr,
        bus_override=limit_bus,
    )

    # Initialize swarm
    particles = np.random.rand(num_particles, dim).astype(float)
    for j in range(dim):
        particles[:, j] = particles[:, j] * (q_maxs[j] - q_mins[j]) + q_mins[j]
    velocities = np.zeros((num_particles, dim), dtype=float)

    pbest = np.copy(particles)
    pbest_fitness = np.full(num_particles, np.inf, dtype=float)

    for i in range(num_particles):
        pbest_fitness[i], _, _, _ = eval_fitness(particles[i])

    gbest_idx = int(np.argmin(pbest_fitness))
    gbest = np.copy(pbest[gbest_idx])
    gbest_fitness = float(pbest_fitness[gbest_idx])

    fitness_history: List[float] = []

    for iter_idx in range(max_iter):
        current_w = float(w - (w - 0.4) * (iter_idx / max_iter)) if max_iter > 0 else float(w)

        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (
                current_w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            max_vel = 0.1 * (q_maxs - q_mins)
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], q_mins, q_maxs)

            current_fitness, _, _, _ = eval_fitness(particles[i])
            if current_fitness < pbest_fitness[i]:
                pbest[i] = np.copy(particles[i])
                pbest_fitness[i] = current_fitness

        current_best_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest = np.copy(pbest[current_best_idx])
            gbest_fitness = float(pbest_fitness[current_best_idx])

        fitness_history.append(float(np.min(pbest_fitness)))

    # Final evaluation (without penalty terms in loss_rate output)
    _, final_loss_rate, final_voltages, power_info = eval_fitness(gbest)
    if final_loss_rate is None or final_voltages is None or power_info is None:
        return {
            "error": "优化后潮流计算不收敛",
            "convergence": False,
            "optimal_params": gbest.tolist(),
            "fitness_history": fitness_history,
            "initial_q_values": initial_q_values,
            "node_names": [node[3] for node in tunable_q_nodes],
            "initial_loss_rate": initial_loss_rate,
            "initial_voltages": initial_voltages.tolist(),
        }

    return {
        "optimal_params": gbest.tolist(),
        "optimal_loss_rate": float(final_loss_rate),
        "initial_loss_rate": float(initial_loss_rate),
        "loss_reduction": float(initial_loss_rate - final_loss_rate),
        "loss_reduction_percent": float((initial_loss_rate - final_loss_rate) / initial_loss_rate * 100.0)
        if initial_loss_rate != 0
        else 0.0,
        "fitness_history": fitness_history,
        "initial_q_values": initial_q_values,
        "node_names": [node[3] for node in tunable_q_nodes],
        "final_voltages": final_voltages.tolist(),
        "initial_voltages": initial_voltages.tolist(),
        "convergence": True,
        "power_info": {
            "balance_node_output": float(power_info[0]),
            "pv_total_injection": float(power_info[1]),
            "total_input_power": float(power_info[2]),
            "total_output_power": float(power_info[3]),
        },
        # Additional diagnostics (safe to ignore for old callers)
        "voltage_bounds_kv": {"v_min": v_bounds.v_min_kv, "v_max": v_bounds.v_max_kv},
        "limit_loss_rate": float(limit_loss_rate) if limit_loss_rate is not None else None,
        "limit_voltages": limit_voltages.tolist() if limit_voltages is not None else None,
    }


def pso_op_v2(*args, **kwargs):
    """Alias for compatibility with potential new imports."""
    return pso_op(*args, **kwargs)


if __name__ == "__main__":
    bus = np.array(
        [
            [1, 0.0, 0.0],
            [2, 1.0, 0.2],
            [3, -0.6, 0.0],
        ],
        dtype=float,
    )
    branch = np.array(
        [
            [1, 1, 2, 0.10, 0.20],
            [2, 2, 3, 0.10, 0.20],
        ],
        dtype=float,
    )
    tunable = [(2, -0.5, 0.5, "PV-3")]
    out = pso_op(bus, branch, tunable, num_particles=10, max_iter=10)
    print(out)
