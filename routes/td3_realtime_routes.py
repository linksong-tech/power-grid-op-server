"""
TD3 实时推理（无功调节策略）路由

接口目标：
- 对齐 .ref/rlCtrl_TD3pure.py 的推理口径：输入关键节点电压(kV) -> 输出 targetQ(MVar)
- 对齐现有项目约定：从线路(lineId)配置读取 keyNodes / 电压上下限 / 可调光伏(调节对象)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from flask import jsonify, request

from lib.line_service import line_service
from td3_core.numpy_models import ActorNetworkNumpy


def _bad_request(message: str, status_code: int = 400):
    return jsonify({"status": "error", "message": message}), status_code


def _load_line_config(line_id: str) -> dict[str, Any]:
    line_data = line_service.get_line(line_id)
    if not line_data:
        raise ValueError("线路不存在")

    key_nodes_data = line_data.get("keyNodesData")
    voltage_limit_data = line_data.get("voltageLimitData")
    adjustable_pv_data = line_data.get("adjustablePvData")

    if not key_nodes_data:
        raise ValueError("该线路缺少关键节点配置(keyNodesData)")
    if not voltage_limit_data or len(voltage_limit_data) != 2:
        raise ValueError("该线路缺少电压上下限配置(voltageLimitData)")
    if not adjustable_pv_data:
        raise ValueError("该线路缺少可调光伏配置(adjustablePvData)")

    try:
        v_min = float(voltage_limit_data[0])
        v_max = float(voltage_limit_data[1])
    except Exception as e:
        raise ValueError(f"电压上下限解析失败: {e}")

    key_node_nos = [str(x) for x in key_nodes_data]
    try:
        key_nodes_idx = [int(node_no) - 1 for node_no in key_node_nos]
    except Exception as e:
        raise ValueError(f"关键节点编号解析失败: {e}")

    tunable_nodes: list[tuple[int, float, float, str, float, str]] = []
    for row in adjustable_pv_data:
        if not row or len(row) < 3:
            continue
        node_no, capacity_mva, device_name = row[0], row[1], row[2]
        try:
            node_idx = int(str(node_no)) - 1
            capacity = float(str(capacity_mva))
            q_max = abs(capacity)
            q_min = -q_max
        except Exception as e:
            raise ValueError(f"可调光伏配置解析失败: {row} ({e})")
        tunable_nodes.append((node_idx, q_min, q_max, str(device_name), capacity, str(node_no)))

    if not tunable_nodes:
        raise ValueError("无有效可调光伏配置")

    return {
        "line_data": line_data,
        "key_node_nos": key_node_nos,
        "key_nodes_idx": key_nodes_idx,
        "voltage_limits_kv": (v_min, v_max),
        "tunable_nodes": tunable_nodes,
    }


def _resolve_model_path_by_line_id(line_id: str, model_path: str) -> str:
    line_dir = line_service._get_line_dir(line_id)
    model_full_path = os.path.join(line_dir, "agent", model_path)
    if not os.path.exists(model_full_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return model_full_path


def _normalize_voltage_kv(voltages_kv: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    normalized = (voltages_kv - v_min) / (v_max - v_min) * 2 - 1
    return np.clip(normalized, -1, 1).astype(np.float32)


def _denormalize_action(
    action: np.ndarray,
    tunable_nodes: list[tuple[int, float, float, str, float, str]],
) -> list[float]:
    q_mins = np.array([node[1] for node in tunable_nodes], dtype=np.float32)
    q_maxs = np.array([node[2] for node in tunable_nodes], dtype=np.float32)

    actual_actions: list[float] = []
    for i in range(len(action)):
        normalized = float(np.clip(action[i], -1, 1))
        actual = (normalized + 1) / 2 * (q_maxs[i] - q_mins[i]) + q_mins[i]
        actual_actions.append(float(actual))
    return actual_actions


def realtime_strategy():
    """
    实时无功调节策略（TD3 推理）

    Request（二选一；同时提供时优先使用 allNodeVoltages）:
    - allNodeVoltages: [{ nodeNo: "1", voltageKv: 10.31 }, ...]
    - keyNodeVoltagesKv: [10.25, 10.30, ...]（顺序需对齐线路 keyNodesData）

    必填:
    - lineId: 线路ID（UUID）
    - modelPath: 模型文件名（.npz，位于该线路目录的 agent/ 下）

    Response:
    - strategy: [{ nodeNo, deviceName, capacityMva, qMinMVar, qMaxMVar, targetQMVar }, ...]
    """
    try:
        data = request.get_json() or {}

        line_id = data.get("lineId")
        model_path = data.get("modelPath")
        if not line_id:
            return _bad_request("缺少必需参数: lineId")
        if not model_path:
            return _bad_request("缺少必需参数: modelPath")

        config = _load_line_config(line_id)
        key_node_nos: list[str] = config["key_node_nos"]
        v_min, v_max = config["voltage_limits_kv"]
        tunable_nodes = config["tunable_nodes"]

        all_node_voltages = data.get("allNodeVoltages")
        key_node_voltages_kv = data.get("keyNodeVoltagesKv")

        mode: str
        if all_node_voltages:
            mode = "allNodeVoltages"
            voltage_map: dict[int, float] = {}
            for item in all_node_voltages:
                if not isinstance(item, dict):
                    return _bad_request("allNodeVoltages 格式错误，应为对象数组")
                if "nodeNo" not in item or "voltageKv" not in item:
                    return _bad_request("allNodeVoltages 元素缺少 nodeNo 或 voltageKv")
                try:
                    node_idx = int(str(item["nodeNo"])) - 1
                    voltage_kv = float(item["voltageKv"])
                except Exception as e:
                    return _bad_request(f"allNodeVoltages 元素解析失败: {item} ({e})")
                voltage_map[node_idx] = voltage_kv

            missing = [node_no for node_no in key_node_nos if (int(node_no) - 1) not in voltage_map]
            if missing:
                return _bad_request(f"缺少关键节点电压: {missing}")

            key_node_voltages = [voltage_map[int(node_no) - 1] for node_no in key_node_nos]
        else:
            mode = "keyNodeVoltagesKv"
            if not isinstance(key_node_voltages_kv, list) or not key_node_voltages_kv:
                return _bad_request("缺少必需参数: allNodeVoltages 或 keyNodeVoltagesKv")
            if len(key_node_voltages_kv) != len(key_node_nos):
                return _bad_request(
                    f"输入电压数量不匹配: 期望{len(key_node_nos)}个，实际{len(key_node_voltages_kv)}个"
                )
            try:
                key_node_voltages = [float(v) for v in key_node_voltages_kv]
            except Exception as e:
                return _bad_request(f"keyNodeVoltagesKv 解析失败: {e}")

        model_full_path = _resolve_model_path_by_line_id(line_id, model_path)

        state_dim = len(key_node_nos)
        action_dim = len(tunable_nodes)
        actor = ActorNetworkNumpy(state_dim=state_dim, action_dim=action_dim, max_action=1.0)
        actor.load(model_full_path)

        normalized_state = _normalize_voltage_kv(np.array(key_node_voltages, dtype=np.float32), v_min, v_max)
        normalized_action = actor.forward(normalized_state.reshape(1, -1), save_cache=False).flatten()
        target_q = _denormalize_action(normalized_action, tunable_nodes)

        strategy = []
        for i, (node_idx, q_min, q_max, device_name, capacity_mva, node_no) in enumerate(tunable_nodes):
            strategy.append(
                {
                    "nodeNo": str(node_no),
                    "deviceName": device_name,
                    "capacityMva": capacity_mva,
                    "qMinMVar": float(q_min),
                    "qMaxMVar": float(q_max),
                    "targetQMVar": float(round(target_q[i], 4)),
                }
            )

        return jsonify(
            {
                "status": "success",
                "data": {
                    "lineId": line_id,
                    "modelPath": model_path,
                    "mode": mode,
                    "voltageLimitsKv": [v_min, v_max],
                    "keyNodes": key_node_nos,
                    "strategy": strategy,
                    "meta": {
                        "stateDim": state_dim,
                        "actionDim": action_dim,
                        "timestamp": data.get("timestamp"),
                    },
                },
            }
        )

    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"实时策略计算失败: {e}"}), 500

