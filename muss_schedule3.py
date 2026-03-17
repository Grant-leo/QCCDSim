# muss_schedule3.py
# ============================================================
# MUSS Scheduler (V3) —— 创新版（接口/结构对齐修复版）
#
# 设计目标：
# 1) 与修复后的 muss_schedule2 在结构、接口、调用、事件语义上保持一致
# 2) 只保留 V3 的内部创新点：
#    - qubit_queues + executed_gate_indices 的 lookahead
#    - rebalance 时的近似 Belady victim 选择
# 3) 修复 V3 中影响 correctness 的严重问题：
#    - 主路由器改为 FreeTrapRoute（capacity-aware）
#    - fire_shuttle 与 route 接口统一
#    - _choose_partition_target 增加 target capacity 过滤
#    - shuttling_direction 增加硬合法性过滤
#    - schedule_gate 增加 route legality 检查
#    - 1Q gate 不再进入 MUSS frontier，改为延后插入
#    - 区分 full_ir 和 twoq-only ir，和 V2 一致
# 4) 保留 shuttle_id 标注，兼容 analyzer aggregate 模式
# ============================================================

import networkx as nx
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *


class MUSSSchedule:
    """
    New paper-faithful interface:
        MUSSSchedule(parse_obj, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)

    Backward-compatible legacy interface:
        MUSSSchedule(ir, gate_info, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)
    """

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
        # 与 muss_schedule2 对齐：同时支持 parse_obj 接口和 legacy 接口
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

        # ------------------------------------------------------
        # 与修复后 V2 对齐：主路径使用 capacity-aware router
        # BasicRoute 仅作调试兜底，不参与主逻辑
        # ------------------------------------------------------
        self.basic_router = BasicRoute(self.machine)
        self.router = None  # sys_state 初始化后绑定为 FreeTrapRoute

        self.gate_finish_times = {}

        # Scheduling statistics
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # ============ shuttle trace / 调试 ============
        self.shuttle_counter = 0
        self.shuttle_log = []
        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None
        self.enable_runtime_trace_print = False
        self.enable_assertions = True
        # ==============================================

        # ------------------------------------------------------
        # 初始化系统状态
        # ------------------------------------------------------
        trap_ions = {}
        seg_ions = {}
        for i in self.machine.traps:
            trap_ions[i.id] = self.init_map[i.id][:] if self.init_map.get(i.id, None) else []
        for i in self.machine.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # 主 router 绑定
        self.router = FreeTrapRoute(self.machine, self.sys_state)

        # 预计算 trap-to-trap 距离缓存
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # ------------------------------------------------------
        # 与 V2 对齐：LRU / protected ions
        # ------------------------------------------------------
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}
        self.protected_ions = set()

        # ------------------------------------------------------
        # 与 V2 对齐：2Q scheduling order
        # V3 创新：在这个 2Q topo order 上建立 qubit_queues / executed set
        # 注意：这里不再混入 full DAG 的 1Q gates
        # ------------------------------------------------------
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)
        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}
        self.gate_to_idx = dict(self.static_topo_order)
        self.gates = self.static_topo_list

        try:
            self.full_topo_list = list(nx.topological_sort(self.full_ir))
        except Exception:
            self.full_topo_list = list(self.full_ir.nodes)
        self.full_topo_order = {g: i for i, g in enumerate(self.full_topo_list)}

        # ------------------------------------------------------
        # V3 创新：只在 2Q-only topo order 上建立 qubit queues
        # 这样不会被 1Q gates 稀释 lookahead horizon
        # ------------------------------------------------------
        self.executed_gate_indices = set()
        self.qubit_queues = collections.defaultdict(list)
        for g_idx, g in enumerate(self.static_topo_list):
            if g in self.gate_info:
                info = self.gate_info[g]
                qs = info if isinstance(info, list) else info["qubits"]
                if len(qs) == 2:
                    for q in qs:
                        self.qubit_queues[q].append(g_idx)

    # ==========================================================
    # trace / debug 输出
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

            steps = rec.get("steps", [])
            for st in steps:
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

    def _trace_print(self, s):
        if self.enable_runtime_trace_print:
            print(s)

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
    # gate payload / ready time / ion location
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
        与修复后 V2 一致：
        2Q MUSS scheduling 的 ready time 只看 twoq-only DAG
        """
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def gate_ready_time_full(self, gate):
        """
        与修复后 V2 一致：
        延后插入 1Q gate 时，用 full DAG 的 ready time
        """
        ready_time = 0
        for in_edge in self.full_ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def ion_ready_info(self, ion_id):
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None

        if len(this_ion_ops):
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]["trap"]
        else:
            did_not_find = True
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    did_not_find = False
                    break
            if did_not_find:
                print("Did not find:", ion_id)
            assert did_not_find is False

        if this_ion_trap != self.sys_state.find_trap_id_by_ion(ion_id):
            print(ion_id, this_ion_trap, self.sys_state.find_trap_id_by_ion(ion_id))
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
    # 基础操作：Split / Move / Merge / Gate
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
        m = self.machine
        move_start = clk

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        # 与修复后 V2 一致：不额外叠加 junction_cross_time
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
    # V3 创新：lookahead
    # 与 V2 结构保持一致，但只替换内部评分逻辑
    # ==========================================================
    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        """
        V3 核心创新：
        - 使用 qubit_queues 只扫描该 ion 真正相关的未来 2Q gates
        - 使用 executed_gate_indices 跳过已执行门
        - 不混入 1Q gates
        """
        score = 0.0
        gamma = 0.9
        weight = 1.0
        lookahead_depth = 20
        found_count = 0

        queue = self.qubit_queues.get(ion, [])

        for g_idx in queue:
            if g_idx in self.executed_gate_indices:
                continue
            if g_idx == current_gate_idx:
                continue
            if found_count >= lookahead_depth:
                break

            gate = self.gates[g_idx]
            if gate in self.gate_info:
                gate_data = self.gate_info[gate]
                q_list = gate_data if isinstance(gate_data, list) else gate_data["qubits"]

                if len(q_list) == 2:
                    partner = q_list[1] if q_list[0] == ion else q_list[0]
                    _, partner_trap = self.ion_ready_info(partner)

                    dist = self.machine.dist_cache.get((target_trap_id, partner_trap), 10)
                    score += weight * dist
                    weight *= gamma
                    found_count += 1

        return score

    def shuttling_direction(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        与修复后 V2 结构一致：
        - 先做 target capacity 硬过滤
        - 非法方向直接剔除
        - 评分内部使用 V3 创新 lookahead
        返回:
          (source_trap, dest_trap)
        若两个方向都非法，返回 (None, None)
        """
        m = self.machine
        ALPHA = 0.7

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

        # Option A: ion1 去 ion2 trap
        cost_move1 = m.dist_cache.get((ion1_trap, ion2_trap), 100)
        future1 = self.get_future_score(ion1, current_gate_idx, ion2_trap)
        future2 = self.get_future_score(ion2, current_gate_idx, ion2_trap)
        score_t2 = cost_move1 + ALPHA * (future1 + future2)

        # Option B: ion2 去 ion1 trap
        cost_move2 = m.dist_cache.get((ion2_trap, ion1_trap), 100)
        future1_t1 = self.get_future_score(ion1, current_gate_idx, ion1_trap)
        future2_t1 = self.get_future_score(ion2, current_gate_idx, ion1_trap)
        score_t1 = cost_move2 + ALPHA * (future1_t1 + future2_t1)

        return (ion1_trap, ion2_trap) if score_t2 < score_t1 else (ion2_trap, ion1_trap)

    # ==========================================================
    # Shuttle：Split / Move / Merge 链
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        与修复后 V2 完全对齐：
        - route 为空时用 capacity-aware route
        - 统一 route 是 Trap/Junction 节点序列
        - identify_start_time + _add_shuttle_ops
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
    # rebalance：结构与 V2 对齐，victim policy 保留 V3 创新
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
        status12, route12 = ftr.find_route(t1, t2)
        status21, route21 = ftr.find_route(t2, t1)

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
        与修复后 V2 结构一致，只替换 victim 选择为 V3 的近似 Belady：
        - 选择 future next-use 最晚 / 永不再用的离子
        - 仍保留 protected_ions 约束
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
                    # -------------------------------
                    # V3 创新：近似 Belady victim
                    # 选“下一次使用最晚 / 不再使用”的离子
                    # -------------------------------
                    best_victim = valid_candidates[0]
                    max_score = -1
                    for ion in valid_candidates:
                        queue = self.qubit_queues.get(ion, [])
                        next_use = float("inf")
                        for g_idx in queue:
                            if g_idx not in self.executed_gate_indices:
                                next_use = g_idx
                                break
                        score = next_use
                        if score > max_score or (score == max_score and ion < best_victim):
                            max_score = score
                            best_victim = ion
                    moving_ion = best_victim

                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)

                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, route=shuttle_route)
                fin_time = max(fin_time, fin_time_new)

        return fin_time

    # ==========================================================
    # meeting target 选择：结构与 V2 对齐，内部评分保留 V3 创新
    # ==========================================================
    def _candidate_meeting_traps(self, trap_a, trap_b):
        if self.is_small_mode:
            return [trap_a, trap_b]

        try:
            ta = self.machine.get_trap(trap_a)
            tb = self.machine.get_trap(trap_b)
        except Exception:
            return [trap_a, trap_b]

        order = {"optical": 0, "operation": 1, "storage": 2}
        a_key = order.get(getattr(ta, "zone_type", "storage"), 9)
        b_key = order.get(getattr(tb, "zone_type", "storage"), 9)

        if a_key < b_key:
            return [trap_a, trap_b]
        if b_key < a_key:
            return [trap_b, trap_a]
        return [trap_a, trap_b]

    def _choose_partition_target(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        与修复后 V2 结构一致：
        - 候选仅限两个 endpoint trap
        - 必须有空位
        - 评分内部保留 V3 创新 lookahead
        返回:
          (moving_ion, target_trap)
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
                move_cost
                + self.get_future_score(ion1, current_gate_idx, target)
                + self.get_future_score(ion2, current_gate_idx, target)
            )

            if best_score is None or score < best_score:
                best = (moving, target)
                best_score = score

        return best

    # ==========================================================
    # Gate scheduling：结构与 V2 完全一致
    # 只保留 V3 的内部 scoring / victim policy 创新
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        与修复后 V2 完全对齐：
        - 这里只调度 2Q gates
        - 1Q gate 不进入 MUSS frontier
        - 1Q gate 由 _schedule_delayed_one_qubit_gates() 统一回填
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
                focus_traps=[ion1_trap, ion2_trap], fire_time=fire_time
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

                # 与修复后 V2 一致：meeting target 合法后仍必须检查 route 合法性
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
    # 延后插入 1Q：与修复后 V2 一致
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
    # local executable helper
    # ==========================================================
    def is_executable_local(self, gate):
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) != 2:
            return False
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # 主循环：与修复后 V2 完全一致
    # 只在 gate 执行完成后更新 V3 的 executed_gate_indices
    # ==========================================================
    def run(self):
        """
        与修复后 V2 完全一致：
        1) MUSS frontier 只跑 2Q-only DAG
        2) local-first
        3) tie-breaking 按 static topo order
        4) 全部 2Q gates 调完后，再按 full DAG 插入 1Q gates
        5) V3 创新：每调完一个 2Q gate，就更新 executed_gate_indices
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
            self.schedule_gate(best_gate, gate_idx=gate_idx)

            # -------------------------------
            # V3 创新：记录已执行 2Q gate index
            # -------------------------------
            self.executed_gate_indices.add(gate_idx)

            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)

        self._schedule_delayed_one_qubit_gates()
