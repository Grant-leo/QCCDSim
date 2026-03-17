# muss_schedule4.py
# ============================================================
# MUSS Scheduler (V4) —— 回退后的轻量鲁棒版
#
# 设计目标：
# 1) 与修复后的 muss_schedule2 在结构、接口、事件语义上保持一致
# 2) 不启用第三方 meeting trap，保持 endpoint-only 会合语义
# 3) 只保留两类轻量创新：
#    - qubit_queues + executed_gate_indices 的轻量 lookahead
#    - rebalance 时更稳健的 victim 选择（next-use + cluster importance）
# 4) 避免上一版 V3 因第三方会合点和重型评分导致的严重 thrashing
# 5) 保持与 analyzer.py、schedule.py、route.py 的接口一致
#
# 重要约束：
# - 不改变 Schedule 事件语义：仍然是 Gate / Split / Move / Merge
# - 不改变 analyzer 依赖的 shuttle_id 标注逻辑
# - 不改变 machine.py 的时间模型接口
# - 不改变 V2 的 1Q 延后插入总结构
# ============================================================

import networkx as nx
import collections
from typing import Dict, List, Set

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *


class MUSSSchedule:
    """
    新接口（推荐）：
        MUSSSchedule(parse_obj, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)

    兼容旧接口：
        MUSSSchedule(ir, gate_info, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)
    """

    # ==========================================================
    # 初始化
    # ==========================================================
    def __init__(
        self,
        ir_or_parse,
        gate_info_or_machine,
        M_or_init_map,
        init_map_or_serial_trap_ops,
        serial_trap_ops=None,
        serial_comm=None,
        global_serial_lock=None,
    ):
        # ------------------------------------------------------
        # 与 muss_schedule2 对齐：支持 parse_obj 接口和 legacy 接口
        # ------------------------------------------------------
        if hasattr(ir_or_parse, "all_gate_map") and hasattr(ir_or_parse, "gate_graph"):
            parse_obj = ir_or_parse
            self.parse_obj = parse_obj
            self.full_ir = parse_obj.gate_graph
            self.ir = getattr(parse_obj, "twoq_gate_graph", parse_obj.gate_graph)
            self.gate_info = parse_obj.all_gate_map
            self.machine = gate_info_or_machine
            self.init_map = M_or_init_map
            self.SerialTrapOps = init_map_or_serial_trap_ops
            self.SerialCommunication = serial_trap_ops
            self.GlobalSerialLock = global_serial_lock
        else:
            self.parse_obj = None
            self.full_ir = ir_or_parse
            self.ir = ir_or_parse
            self.gate_info = gate_info_or_machine
            self.machine = M_or_init_map
            self.init_map = init_map_or_serial_trap_ops
            self.SerialTrapOps = serial_trap_ops
            self.SerialCommunication = serial_comm
            self.GlobalSerialLock = global_serial_lock

        self.architecture_scale = str(
            getattr(self.machine.mparams, "architecture_scale", "small")
        ).lower()
        self.is_small_mode = self.architecture_scale == "small"
        self.is_large_mode = not self.is_small_mode

        self.schedule = Schedule(self.machine)

        # 主路径使用 capacity-aware route；BasicRoute 仅保留为调试兜底
        self.basic_router = BasicRoute(self.machine)
        self.router = None  # sys_state 初始化后绑定为 FreeTrapRoute

        self.gate_finish_times = {}

        # 统计量
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # shuttle trace
        self.shuttle_counter = 0
        self.shuttle_log = []
        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        self.enable_runtime_trace_print = False
        self.enable_assertions = True

        # ------------------------------------------------------
        # 初始化系统状态
        # ------------------------------------------------------
        trap_ions = {}
        seg_ions = {}
        for t in self.machine.traps:
            trap_ions[t.id] = self.init_map[t.id][:] if self.init_map.get(t.id, None) else []
        for s in self.machine.segments:
            seg_ions[s.id] = []

        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # 绑定主路由器
        self.router = FreeTrapRoute(self.machine, self.sys_state)

        # 预计算 trap-to-trap 距离
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # LRU / 最近使用时间
        all_ions = set()
        for ions in trap_ions.values():
            all_ions.update(ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # 当前 gate 保护离子
        self.protected_ions = set()

        # ------------------------------------------------------
        # 两套拓扑序：
        # 1) static_topo_list: 2Q-only DAG，用于 MUSS frontier
        # 2) full_topo_list:  full DAG，用于 1Q 延后插入
        # ------------------------------------------------------
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)

        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}
        self.gates = self.static_topo_list

        try:
            self.full_topo_list = list(nx.topological_sort(self.full_ir))
        except Exception:
            self.full_topo_list = list(self.full_ir.nodes)

        self.full_topo_order = {g: i for i, g in enumerate(self.full_topo_list)}

        # ------------------------------------------------------
        # V3 创新：只在 2Q-only topo order 上建立 qubit queues
        # 避免被 1Q gate 稀释 lookahead
        # ------------------------------------------------------
        self.executed_gate_indices: Set[int] = set()
        self.qubit_queues: Dict[int, List[int]] = collections.defaultdict(list)

        for g_idx, g in enumerate(self.static_topo_list):
            if g in self.gate_info:
                info = self.gate_info[g]
                qs = info if isinstance(info, list) else info["qubits"]
                if len(qs) == 2:
                    for q in qs:
                        self.qubit_queues[q].append(g_idx)

        # ------------------------------------------------------
        # 轻量 V3 参数：
        # 只保留温和的未来代价和负载惩罚，不再使用重型 hotspot / eviction / freeze
        # ------------------------------------------------------
        self.lookahead_depth = 8
        self.lookahead_gamma = 0.9

        # target 评分项
        self.w_move = 1.0
        self.w_future = 0.6
        self.w_load = 0.8

        # victim 评分项
        self.victim_w_next_use = 1.0
        self.victim_w_cluster = 1.2

        # 当前 gate 序号（用于 lookahead / victim 打分）
        self.gate_step_counter = 0

    # ==========================================================
    # 调试 / trace
    # ==========================================================
    def dump_shuttle_trace(self, max_lines=None):
        lines = []
        for rec in self.shuttle_log:
            sid = rec.get("shuttle_id")
            ion = rec.get("ion")
            src = rec.get("src_trap")
            dst = rec.get("dst_trap")
            route_txt = rec.get("route_text", "")
            lines.append(f"[SHUTTLE {sid}] ion={ion}  T{src} -> T{dst}  route={route_txt}")

            for st in rec.get("steps", []):
                et = st["etype"]
                stt = st["t_start"]
                edt = st["t_end"]
                desc = st["desc"]
                lines.append(f"    - {et:<5}  ({stt} -> {edt})  {desc}")

        if max_lines is not None:
            lines = lines[:max_lines]
        return "\n".join(lines)

    def dump_schedule_events(self):
        self.schedule.print_events()

    def _trace_print(self, msg: str):
        if self.enable_runtime_trace_print:
            print(msg)

    def _trace_add_step(self, etype, t_start, t_end, desc):
        if self._current_shuttle_id is None:
            return
        sid = self._current_shuttle_id
        if sid < 0 or sid >= len(self.shuttle_log):
            return
        self.shuttle_log[sid]["steps"].append(
            {"etype": etype, "t_start": int(t_start), "t_end": int(t_end), "desc": desc}
        )

    def _annotate_last_event_with_shuttle_id(self):
        """
        给最后一个 Schedule 事件打上 shuttle_id。
        analyzer 在 aggregate shuttle fidelity 模式下依赖这个字段。
        """
        if self._current_shuttle_id is None:
            return
        if not hasattr(self.schedule, "events"):
            return
        if not self.schedule.events:
            return
        try:
            self.schedule.events[-1][4]["shuttle_id"] = self._current_shuttle_id
        except Exception:
            pass

    # ==========================================================
    # 基础查询：gate payload / ready time / ion location
    # ==========================================================
    def _gate_payload(self, gate):
        data = self.gate_info.get(gate, None)
        if data is None:
            return None, [], "unknown"
        if isinstance(data, dict):
            return data, list(data.get("qubits", [])), data.get("type", "unknown")
        return {"qubits": list(data), "type": "cx" if len(data) == 2 else "u"}, list(data), "cx" if len(data) == 2 else "u"

    def gate_ready_time(self, gate):
        """
        2Q MUSS scheduling 的 ready time 只看 2Q-only DAG。
        """
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def gate_ready_time_full(self, gate):
        """
        延后插入 1Q gate 时，用 full DAG 的 ready time。
        """
        ready_time = 0
        for in_edge in self.full_ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def ion_ready_info(self, ion_id):
        """
        返回：
            (该 ion 最近一次操作完成时间, 当前所在 trap_id)
        并与 sys_state 做一致性检查。
        """
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None

        if len(this_ion_ops):
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]["trap"]
        else:
            found = False
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    found = True
                    break
            if not found:
                raise AssertionError(f"Did not find ion {ion_id} in init_map")

        sys_trap = self.sys_state.find_trap_id_by_ion(ion_id)
        if this_ion_trap != sys_trap:
            print("ion location mismatch:", ion_id, this_ion_trap, sys_trap)
            self.sys_state.print_state()
            raise AssertionError("ion location mismatch between schedule-inferred and sys_state")

        return this_ion_last_op_time, this_ion_trap

    # ==========================================================
    # 容量 / 路由辅助
    # ==========================================================
    def _trap_has_free_slot(self, trap_id, incoming=1):
        cur = len(self.sys_state.trap_ions[trap_id])
        cap = self.machine.traps[trap_id].capacity
        return (cur + incoming) <= cap

    def _find_route_or_none(self, source_trap, dest_trap):
        status, route = self.router.find_route(source_trap, dest_trap)
        if status == 0:
            return route
        return None

    # ==========================================================
    # 事件级基本操作：Split / Move / Merge / Gate
    # ==========================================================
    def add_split_op(self, clk, src_trap, dest_seg, ion):
        m = self.machine

        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(src_trap.id)
            split_start = max(clk, last_event_time_on_trap)
        else:
            split_start = clk

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            split_start = max(split_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            split_start = max(split_start, last_event_time_in_system)

        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = \
            m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)

        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration

        self.schedule.add_split_or_merge(
            split_start, split_end, [ion],
            src_trap.id, dest_seg.id,
            Schedule.Split,
            split_swap_count, split_swap_hops, i1, i2, ion_swap_hops
        )

        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step(
            "SPLIT", split_start, split_end,
            f"ion {ion}: T{src_trap.id} -> Seg{dest_seg.id} "
            f"(swap_cnt={split_swap_count}, swap_hops={split_swap_hops}, ion_hops={ion_swap_hops}, i1={i1}, i2={i2})"
        )
        self._trace_print(f"[TRACE] SPLIT ion={ion} T{src_trap.id} -> Seg{dest_seg.id} ({split_start}->{split_end})")

        return split_end

    def add_merge_op(self, clk, dest_trap, src_seg, ion):
        m = self.machine

        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(dest_trap.id)
            merge_start = max(clk, last_event_time_on_trap)
        else:
            merge_start = clk

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            merge_start = max(merge_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            merge_start = max(merge_start, last_event_time_in_system)

        merge_end = merge_start + m.merge_time(dest_trap.id)

        self.schedule.add_split_or_merge(
            merge_start, merge_end, [ion],
            dest_trap.id, src_seg.id,
            Schedule.Merge,
            0, 0, 0, 0, 0
        )

        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step("MERGE", merge_start, merge_end, f"ion {ion}: Seg{src_seg.id} -> T{dest_trap.id}")
        self._trace_print(f"[TRACE] MERGE ion={ion} Seg{src_seg.id} -> T{dest_trap.id} ({merge_start}->{merge_end})")

        return merge_end

    def add_move_op(self, clk, src_seg, dest_seg, junct, ion):
        """
        与修复后的 V2 保持一致：
        junction 只作为调度冲突资源，不额外叠加物理时延。
        """
        m = self.machine
        move_start = clk

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        move_end = move_start + m.move_time(src_seg.id, dest_seg.id)
        move_start, move_end = self.schedule.junction_traffic_crossing(src_seg, dest_seg, junct, move_start, move_end)

        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)
        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step(
            "MOVE", move_start, move_end,
            f"ion {ion}: Seg{src_seg.id} -> Seg{dest_seg.id} via J{junct.id}"
        )
        self._trace_print(f"[TRACE] MOVE ion={ion} Seg{src_seg.id}->{dest_seg.id} via J{junct.id} ({move_start}->{move_end})")

        return move_end

    def add_gate_op(self, clk, trap_id, gate, ion1, ion2):
        fire_time = clk

        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(trap_id)
            fire_time = max(clk, last_event_time_on_trap)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            fire_time = max(fire_time, last_event_time_in_system)

        gate_duration = self.machine.gate_time(self.sys_state, trap_id, ion1, ion2)
        self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], trap_id)
        self.gate_finish_times[gate] = fire_time + gate_duration
        return fire_time + gate_duration

    # ==========================================================
    # V3 轻量创新一：future score
    # ==========================================================
    def _future_partner_info(self, ion, current_gate_idx, horizon=None):
        """
        返回该 ion 在未来若干个 2Q gate 中的 partner 列表。
        """
        if horizon is None:
            horizon = self.lookahead_depth

        partners = []
        queue = self.qubit_queues.get(ion, [])

        for g_idx in queue:
            if g_idx in self.executed_gate_indices:
                continue
            if g_idx == current_gate_idx:
                continue
            if len(partners) >= horizon:
                break

            gate = self.gates[g_idx]
            if gate in self.gate_info:
                gate_data = self.gate_info[gate]
                q_list = gate_data if isinstance(gate_data, list) else gate_data["qubits"]
                if len(q_list) == 2:
                    partner = q_list[1] if q_list[0] == ion else q_list[0]
                    partners.append((g_idx, partner))
        return partners

    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        """
        温和 lookahead：
        只看未来少量 2Q gate，并对 partner 到 target_trap 的距离做折扣累加。
        """
        score = 0.0
        weight = 1.0
        partners = self._future_partner_info(ion, current_gate_idx, self.lookahead_depth)

        for _, partner in partners:
            _, partner_trap = self.ion_ready_info(partner)
            dist = self.machine.dist_cache.get((target_trap_id, partner_trap), 10)
            score += weight * dist
            weight *= self.lookahead_gamma

        return score

    def _target_load_penalty(self, trap_id):
        """
        轻量负载惩罚：
        trap 越满，越不鼓励继续把离子搬进去。
        """
        cur = len(self.sys_state.trap_ions[trap_id])
        cap = self.machine.traps[trap_id].capacity
        if cap <= 0:
            return 1e9
        return float(cur) / float(cap)

    # ==========================================================
    # V3 轻量创新二：victim score
    # ==========================================================
    def _next_use_index(self, ion):
        """
        返回该 ion 下一次在 2Q-only topo order 中出现的位置。
        """
        queue = self.qubit_queues.get(ion, [])
        for g_idx in queue:
            if g_idx not in self.executed_gate_indices:
                return g_idx
        return float("inf")

    def _cluster_importance(self, ion, trap_id, current_gate_idx):
        """
        估计该 ion 在当前 trap 内是否是“局部簇”的核心。
        如果未来伙伴很多就在当前 trap 或 1-hop trap，则不应轻易迁出。
        """
        score = 0.0
        future = self._future_partner_info(ion, current_gate_idx, horizon=6)

        for _, partner in future:
            _, p_trap = self.ion_ready_info(partner)
            d = self.machine.dist_cache.get((trap_id, p_trap), 10)
            if d == 0:
                score += 2.0
            elif d == 1:
                score += 1.0

        return score

    def _victim_score(self, ion, trap_id, current_gate_idx):
        """
        victim 分数越大，越适合被迁出。
        这里只保留两项：
        - 下次使用越晚，越适合迁出
        - cluster importance 越高，越不适合迁出
        """
        next_use = self._next_use_index(ion)
        if next_use == float("inf"):
            next_use_term = 1000.0
        else:
            next_use_term = float(next_use)

        cluster_term = self._cluster_importance(ion, trap_id, current_gate_idx)

        score = (
            self.victim_w_next_use * next_use_term
            - self.victim_w_cluster * cluster_term
        )
        return score

    # ==========================================================
    # endpoint-only 会合点候选
    # ==========================================================
    def _candidate_meeting_traps(self, trap_a, trap_b):
        """
        回退后的轻量鲁棒版 V3：
        small mode 与 large mode 都先保持 endpoint-only。
        这样不会改变 V2 的会合物理语义，只在“选哪边会合”上做轻量创新。
        """
        return [trap_a, trap_b]

    def _choose_partition_target(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        返回：
            (moving_ion, target_trap)

        结构与 V2 保持一致，只增强内部评分：
            move_cost + 轻量 future_score + 轻量 load_penalty
        """
        candidates = self._candidate_meeting_traps(ion1_trap, ion2_trap)
        best = None
        best_score = None

        for target in candidates:
            if not self._trap_has_free_slot(target, incoming=1):
                continue

            if target == ion1_trap:
                moving = ion2
                move_cost = self.machine.trap_distance(ion2_trap, target)
            elif target == ion2_trap:
                moving = ion1
                move_cost = self.machine.trap_distance(ion1_trap, target)
            else:
                continue

            score = (
                self.w_move * move_cost
                + self.w_future * (
                    self.get_future_score(ion1, current_gate_idx, target)
                    + self.get_future_score(ion2, current_gate_idx, target)
                )
                + self.w_load * self._target_load_penalty(target)
            )

            if best_score is None or score < best_score:
                best = (moving, target)
                best_score = score

        return best

    # ==========================================================
    # fallback 方向选择：仍只在 endpoint 间回退
    # ==========================================================
    def shuttling_direction(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        若 _choose_partition_target() 没给出结果，则在两个 endpoint 之间回退。
        """
        legal_meet_at_t1 = self._trap_has_free_slot(ion1_trap, incoming=1)
        legal_meet_at_t2 = self._trap_has_free_slot(ion2_trap, incoming=1)

        if not legal_meet_at_t1 and not legal_meet_at_t2:
            return None, None
        if legal_meet_at_t1 and not legal_meet_at_t2:
            return ion2_trap, ion1_trap
        if legal_meet_at_t2 and not legal_meet_at_t1:
            return ion1_trap, ion2_trap

        if current_gate_idx is None:
            return ion1_trap, ion2_trap

        score_t2 = (
            self.machine.dist_cache.get((ion1_trap, ion2_trap), 100)
            + self.w_future * (
                self.get_future_score(ion1, current_gate_idx, ion2_trap)
                + self.get_future_score(ion2, current_gate_idx, ion2_trap)
            )
            + self.w_load * self._target_load_penalty(ion2_trap)
        )

        score_t1 = (
            self.machine.dist_cache.get((ion2_trap, ion1_trap), 100)
            + self.w_future * (
                self.get_future_score(ion1, current_gate_idx, ion1_trap)
                + self.get_future_score(ion2, current_gate_idx, ion1_trap)
            )
            + self.w_load * self._target_load_penalty(ion1_trap)
        )

        return (ion1_trap, ion2_trap) if score_t2 < score_t1 else (ion2_trap, ion1_trap)

    # ==========================================================
    # Shuttle 事件链
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        与修复后的 V2 完全对齐：
        - 若 route 为空，则使用 capacity-aware route
        - 先估时，再 identify_start_time
        - 再写入 split / move / merge
        """
        m = self.machine

        if len(route):
            rpath = route
        else:
            rpath = self._find_route_or_none(src_trap, dest_trap)
            if rpath is None:
                raise RuntimeError(f"No legal route found for shuttle: T{src_trap} -> T{dest_trap}")

        shuttle_id = self.shuttle_counter
        self.shuttle_counter += 1

        self._current_shuttle_id = shuttle_id
        self._current_shuttle_route = rpath
        self._current_shuttle_ion = ion
        self._current_shuttle_src = src_trap
        self._current_shuttle_dst = dest_trap

        route_txt = []
        for node in rpath:
            if isinstance(node, Trap):
                route_txt.append(f"T{node.id}")
            elif isinstance(node, Junction):
                route_txt.append(f"J{node.id}")
            else:
                route_txt.append(str(node))

        src_id = src_trap.id if isinstance(src_trap, Trap) else int(src_trap)
        dst_id = dest_trap.id if isinstance(dest_trap, Trap) else int(dest_trap)

        self.shuttle_log.append(
            {
                "shuttle_id": shuttle_id,
                "ion": ion,
                "src_trap": src_id,
                "dst_trap": dst_id,
                "route_text": "->".join(route_txt),
                "steps": [],
            }
        )
        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} START ion={ion} T{src_id}->{dst_id} route={route_txt} ===")

        t_est = 0
        for i in range(len(rpath) - 1):
            u = rpath[i]
            v = rpath[i + 1]
            seg = self.machine.graph[u][v]["seg"]

            if isinstance(u, Trap) and isinstance(v, Junction):
                t_est += m.mparams.split_merge_time
            elif isinstance(u, Junction) and isinstance(v, Junction):
                t_est += m.move_time(seg.id, seg.id)
            elif isinstance(u, Junction) and isinstance(v, Trap):
                t_est += m.merge_time(v.id)

        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)
        clk = self._add_shuttle_ops(rpath, ion, clk)

        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} END at t={clk} ===")

        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        return clk

    def _add_shuttle_ops(self, spath, ion, clk):
        trap_pos = []
        for i in range(len(spath)):
            if type(spath[i]) == Trap:
                trap_pos.append(i)

        for i in range(len(trap_pos) - 1):
            idx0 = trap_pos[i]
            idx1 = trap_pos[i + 1] + 1

            clk = self._add_partial_shuttle_ops(spath[idx0:idx1], ion, clk)

            self.sys_state.trap_ions[spath[trap_pos[i]].id].remove(ion)

            last_junct = spath[trap_pos[i + 1] - 1]
            dest_trap = spath[trap_pos[i + 1]]
            last_seg = self.machine.graph[last_junct][dest_trap]["seg"]
            orient = dest_trap.orientation[last_seg.id]
            if orient == "R":
                self.sys_state.trap_ions[dest_trap.id].append(ion)
            else:
                self.sys_state.trap_ions[dest_trap.id].insert(0, ion)

        return clk

    def _add_partial_shuttle_ops(self, spath, ion, clk):
        assert len([item for item in spath if type(item) == Trap]) == 2

        seg_list = []
        for i in range(len(spath) - 1):
            u = spath[i]
            v = spath[i + 1]
            seg_list.append(self.machine.graph[u][v]["seg"])

        clk = self.add_split_op(clk, spath[0], seg_list[0], ion)

        for i in range(len(seg_list) - 1):
            u = seg_list[i]
            v = seg_list[i + 1]
            junct = spath[1 + i]
            clk = self.add_move_op(clk, u, v, junct, ion)

        clk = self.add_merge_op(clk, spath[-1], seg_list[-1], ion)
        return clk

    # ==========================================================
    # rebalance：结构与 V2 一致，victim policy 轻量增强
    # ==========================================================
    def rebalance_traps(self, focus_traps, fire_time):
        m = self.machine
        ss = self.sys_state
        t1 = focus_traps[0]
        t2 = focus_traps[1]
        excess_cap1 = m.traps[t1].capacity - len(ss.trap_ions[t1])
        excess_cap2 = m.traps[t2].capacity - len(ss.trap_ions[t2])
        need_rebalance = False

        ftr = FreeTrapRoute(m, ss)
        status12, _ = ftr.find_route(t1, t2)
        status21, _ = ftr.find_route(t2, t1)

        if excess_cap1 == 0 and excess_cap2 == 0:
            need_rebalance = True
        else:
            if status12 == 1 and status21 == 1:
                need_rebalance = True

        if need_rebalance:
            finish_time = self.do_rebalance_traps(fire_time)
            return 1, finish_time
        else:
            return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        """
        保持 V2 的 flow-based clear-all-blocks 框架，
        但 victim 评分改为：
            next-use + cluster importance
        """
        self.count_rebalance += 1
        rebal = RebalanceTraps(self.machine, self.sys_state)
        flow_dict = rebal.clear_all_blocks()

        shuttle_graph = nx.DiGraph()
        used_flow = {}
        for i in flow_dict:
            for j in flow_dict[i]:
                if flow_dict[i][j] != 0:
                    shuttle_graph.add_edge(i, j, weight=flow_dict[i][j])
                    used_flow[(i, j)] = 0

        fin_time = fire_time
        current_gate_idx = self.gate_step_counter

        for node in shuttle_graph.nodes():
            if shuttle_graph.in_degree(node) == 0 and type(node) == Trap:
                updated_graph = shuttle_graph.copy()
                for edge in list(used_flow.keys()):
                    if edge in updated_graph.edges:
                        if used_flow[edge] == updated_graph[edge[0]][edge[1]]["weight"]:
                            updated_graph.remove_edge(edge[0], edge[1])

                T = nx.dfs_tree(updated_graph, source=node)
                shuttle_route = []

                for tnode in T:
                    if T.out_degree(tnode) == 0 and tnode != node:
                        try:
                            shuttle_route = nx.shortest_path(T, node, tnode)
                            break
                        except Exception:
                            continue

                if not shuttle_route:
                    continue

                for i in range(len(shuttle_route) - 1):
                    e0 = shuttle_route[i]
                    e1 = shuttle_route[i + 1]
                    if (e0, e1) in used_flow:
                        used_flow[(e0, e1)] += 1
                    elif (e1, e0) in used_flow:
                        used_flow[(e1, e0)] += 1

                candidates = self.sys_state.trap_ions[node.id]
                valid_candidates = [ion for ion in candidates if ion not in self.protected_ions]

                if not valid_candidates:
                    moving_ion = candidates[0]
                else:
                    best_victim = None
                    best_score = None
                    for ion in valid_candidates:
                        sc = self._victim_score(ion, node.id, current_gate_idx)
                        if best_score is None or sc > best_score:
                            best_score = sc
                            best_victim = ion
                    moving_ion = best_victim

                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)

                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, route=shuttle_route)
                fin_time = max(fin_time, fin_time_new)

        return fin_time

    # ==========================================================
    # Gate scheduling：与 V2 结构保持一致
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        与修复后的 V2 对齐：
        - 这里只调度 2Q gates
        - 1Q gates 延后插入
        - 会合语义保持 endpoint-only
        - 仅在“选哪边会合”“堵了后赶走谁”上做轻量创新
        """
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)
            return

        if len(qubits) != 2:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)
            return

        ion1, ion2 = qubits[0], qubits[1]
        self.protected_ions = {ion1, ion2}

        ready = self.gate_ready_time(gate)
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        fire_time = max(ready, ion1_time, ion2_time, specified_time)

        finish_time = 0
        if ion1_trap == ion2_trap:
            gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
            zone_type = getattr(self.machine.get_trap(ion1_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(
                fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap,
                gate_type=gate_type, zone_type=zone_type, gate_id=gate
            )
            self.gate_finish_times[gate] = fire_time + gate_duration
            finish_time = fire_time + gate_duration
        else:
            rebal_flag, new_fin_time = self.rebalance_traps(
                focus_traps=[ion1_trap, ion2_trap],
                fire_time=fire_time
            )

            if not rebal_flag:
                meet_choice = self._choose_partition_target(
                    ion1_trap, ion2_trap, ion1, ion2, gate_idx
                )

                if meet_choice is not None:
                    moving_ion, dest_trap = meet_choice
                    source_trap = ion1_trap if moving_ion == ion1 else ion2_trap
                else:
                    source_trap, dest_trap = self.shuttling_direction(
                        ion1_trap, ion2_trap, ion1, ion2, gate_idx
                    )
                    if source_trap is None or dest_trap is None:
                        self.protected_ions = set()
                        self.schedule_gate(
                            gate,
                            specified_time=self.do_rebalance_traps(fire_time),
                            gate_idx=gate_idx
                        )
                        return
                    moving_ion = ion1 if source_trap == ion1_trap else ion2

                route = self._find_route_or_none(source_trap, dest_trap)
                if route is None:
                    self.protected_ions = set()
                    self.schedule_gate(
                        gate,
                        specified_time=self.do_rebalance_traps(fire_time),
                        gate_idx=gate_idx
                    )
                    return

                clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time, route=route)

                dest_ions = self.sys_state.trap_ions[dest_trap]
                if ion1 not in dest_ions or ion2 not in dest_ions:
                    raise RuntimeError(
                        f"2Q gate cannot execute on trap {dest_trap}: ions not co-located. "
                        f"gate={gate}, ion1={ion1}, ion2={ion2}, trap_ions={dest_ions}, "
                        f"source_trap={source_trap}, moving_ion={moving_ion}"
                    )

                gate_duration = self.machine.gate_time(self.sys_state, dest_trap, ion1, ion2)
                zone_type = getattr(self.machine.get_trap(dest_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
                self.schedule.add_gate(
                    clk, clk + gate_duration, [ion1, ion2], dest_trap,
                    gate_type=gate_type, zone_type=zone_type, gate_id=gate
                )
                self.gate_finish_times[gate] = clk + gate_duration
                finish_time = clk + gate_duration
            else:
                self.protected_ions = set()
                self.schedule_gate(gate, specified_time=new_fin_time, gate_idx=gate_idx)
                return

        self.ion_last_used[ion1] = finish_time
        self.ion_last_used[ion2] = finish_time
        self.protected_ions = set()

    # ==========================================================
    # 1Q gate 延后插入（与 V2 对齐）
    # ==========================================================
    def add_one_qubit_gate(self, gate):
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None or len(qubits) != 1:
            return

        ion = qubits[0]
        ready = self.gate_ready_time_full(gate)
        ion_time, ion_trap = self.ion_ready_info(ion)
        fire_time = max(ready, ion_time)

        duration = self.machine.single_qubit_gate_time(gate_type)
        zone_type = getattr(self.machine.get_trap(ion_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
        self.schedule.add_gate(
            fire_time, fire_time + duration, [ion], ion_trap,
            gate_type=gate_type, zone_type=zone_type, gate_id=gate
        )
        self.gate_finish_times[gate] = fire_time + duration
        self.ion_last_used[ion] = fire_time + duration

    def _schedule_delayed_one_qubit_gates(self):
        if self.full_ir is None:
            return
        for g in self.full_topo_list:
            gate_data, qubits, _ = self._gate_payload(g)
            if gate_data is None:
                continue
            if len(qubits) == 1:
                self.add_one_qubit_gate(g)

    # ==========================================================
    # local helper
    # ==========================================================
    def is_executable_local(self, gate):
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) != 2:
            return False
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # 主循环：与 V2 对齐，但执行后更新 V3 状态
    # ==========================================================
    def run(self):
        """
        1) MUSS frontier 只运行在 2Q-only DAG 上
        2) local-first
        3) 全部 2Q gate 完成后，再按 full DAG 插入 1Q gates
        4) 每调度一个 2Q gate，更新 executed_gate_indices 和 gate_step_counter
        """
        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}
        ready_gates = [n for n in self.ir.nodes if in_degree[n] == 0]

        processed_count = 0
        total_gates = len(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_gates:
                break

            local_candidates = []
            remote_candidates = []

            for g in ready_gates:
                if self.is_executable_local(g):
                    local_candidates.append(g)
                else:
                    remote_candidates.append(g)

            if local_candidates:
                best_gate = min(local_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))
            else:
                best_gate = min(remote_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))

            gate_idx = self.static_topo_order.get(best_gate, 0)
            self.gate_step_counter = gate_idx

            self.schedule_gate(best_gate, gate_idx=gate_idx)

            self.executed_gate_indices.add(gate_idx)

            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)

        self._schedule_delayed_one_qubit_gates()
