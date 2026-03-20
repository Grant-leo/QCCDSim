# muss_schedule2.py
# ============================================================
# MUSS Scheduler (V6) —— 严格按论文路径的局部冲突处理版
#
# 设计目标：
# 1) 严格保留论文主干：frontier + prioritize executable gates + multi-level scheduling + LRU
# 2) 按论文“冲突处理”路径实现局部逐步驱逐，而不是使用全局 flow rebalance
# 3) 保留 target trap capacity 检查与 capacity-aware route 修复
# 4) 保留 shuttle_id 标注，供 analyzer 聚合同一次 shuttle 的保真度
# 5) 保留 1Q 延后插入逻辑，保证 timing / fidelity replay 正确
# 6) 显式去除 look-ahead / future-score 相关评分，避免把论文 3.3 的思想混入小规模 3.2 调度
# 7) 除调度冲突处理路径外，其它接口和既有行为尽量不改
#
# 说明：
# - 论文第 3.2 节的冲突处理强调：当目标 zone 被占满时，采用 multi-level scheduling
#   选择新的安置 zone，并用 LRU 驱逐长期未使用离子，而不是做全局网络流式 rebalance。
# - 因此 V6 在 V5 基础上移除“遇到死锁直接报错”的做法，改成“局部、递进、可证明合法”的
#   conflict handling：先定目标执行阱，再为其腾挪空间，然后再执行当前 2Q gate。
# ============================================================

import networkx as nx
import numpy as np
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction

# 注意：
# 这里故意不再导入 rebalance.py。
# 这是本文件最关键的“干净消融”改动之一。
# from rebalance import *


class MUSSSchedule:
    """
    输入:
      1) ir: gate dependency DAG (networkx DiGraph)
      2) gate_info: gate -> involved qubits（可能是 list，也可能是 dict{qubits/type/...}）
      3) M: machine object
      4) init_map: 初始映射 trap_id -> [ion_ids...]（链顺序）
      5) 串行开关：SerialTrapOps / SerialCommunication / GlobalSerialLock
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
        """
        新接口（论文复现主接口）:
            MUSSSchedule(parse_obj, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)

        兼容旧接口:
            MUSSSchedule(ir, gate_info, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)
        """
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

        # 主路径使用 capacity-aware router；BasicRoute 仅保留作调试兜底
        self.basic_router = BasicRoute(self.machine)
        self.router = None  # 在 sys_state 初始化之后绑定为 FreeTrapRoute
        self.gate_finish_times = {}

        # 调度统计信息
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # ============ 可观测性（调度正确性验证用） ============
        self.shuttle_counter = 0
        self.shuttle_log = []

        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        self.enable_runtime_trace_print = False
        # =====================================================

        # -------- 初始化系统状态 MachineState --------
        trap_ions = {}
        seg_ions = {}
        for i in self.machine.traps:
            trap_ions[i.id] = self.init_map[i.id][:] if self.init_map.get(i.id, None) else []
        for i in self.machine.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # 绑定 capacity-aware router
        self.router = FreeTrapRoute(self.machine, self.sys_state)

        # 预计算 trap-to-trap 最短路（给 endpoint 目标选择与局部驱逐排序使用）
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === MUSS Strict Requirement: LRU Tracking ===
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # 保护“当前 gate 涉及的离子”不被驱逐
        # 注意：在本无-rebalance版本中，这个集合仍保留，
        # 因为它属于原有调度语义的一部分，也便于后续一致性扩展。
        self.protected_ions = set()

        # 2Q scheduling order (paper-faithful MUSS)
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)
        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}
        self.gates = self.static_topo_list

        # Full-program order kept for delayed 1Q scheduling / timing replay
        try:
            self.full_topo_list = list(nx.topological_sort(self.full_ir))
        except Exception:
            self.full_topo_list = list(self.full_ir.nodes)
        self.full_topo_order = {g: i for i, g in enumerate(self.full_topo_list)}

    # ==========================================================
    # 观测/导出接口
    # ==========================================================
    def dump_shuttle_trace(self, max_lines=None):
        """
        输出所有 shuttle 的全过程记录：
          shuttle_id / ion / src->dst / route / split/move/merge 时间段
        """
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
        """直接打印 Schedule.events（粗粒度），便于和 analyzer replay 对照。"""
        self.schedule.print_events()

    def _trace_print(self, s):
        if self.enable_runtime_trace_print:
            print(s)

    def _trace_add_step(self, etype, t_start, t_end, desc):
        """
        在当前 shuttle 上下文里追加一个 step。
        只有在 shuttle 过程中才会记录（_current_shuttle_id != None）
        """
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
        给刚刚写入 schedule 的最后一个事件补充 shuttle_id。
        这样 analyzer 在 aggregate 模式下才能把
        Split / Move / Merge 聚合成同一次 shuttle。
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
    # Ready time / ion location 推断
    # ==========================================================
    def gate_ready_time(self, gate):
        """根据 2Q-only DAG 依赖边，找到 gate 最早可执行时间（给 MUSS 2Q 调度使用）。"""
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def gate_ready_time_full(self, gate):
        """根据完整 DAG 依赖边，找到 gate 最早可执行时间（给延后插入的 1Q gate 使用）。"""
        ready_time = 0
        for in_edge in self.full_ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def ion_ready_info(self, ion_id):
        """
        返回 (该 ion 最近一次操作完成时间, 当前所在 trap_id)。
        并做一致性检查：schedule 推断的位置必须与 sys_state 一致。
        """
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None

        if len(this_ion_ops):
            # 最后一次必须是 Gate 或 Merge（因为 Split/Move 结束后应 Merge 回 trap 才能 gate）
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]["trap"]
        else:
            # 没有历史事件：从 init_map 里找
            did_not_find = True
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    did_not_find = False
                    break
            if did_not_find:
                print("Did not find:", ion_id)
            assert did_not_find is False

        # 强一致性检查：schedule 推断位置 vs sys_state
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
        """
        统一包装 FreeTrapRoute：
          - 返回合法路径则给出 route
          - 若被 block，则返回 None
        """
        status, route = self.router.find_route(source_trap, dest_trap)
        if status == 0:
            return route
        return None

    # ==========================================================
    # 基础操作：Split / Move / Merge / Gate
    # ==========================================================
    def add_split_op(self, clk, src_trap, dest_seg, ion):
        """
        在 src_trap 上把 ion split 到 dest_seg。
        会考虑：
          - Trap 串行（SerialTrapOps）
          - Comm 串行（SerialCommunication）
          - 全局串行（GlobalSerialLock）
        """
        m = self.machine

        # 1) 决定 split_start
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

        # 2) 计算 split 时间 + swap 信息
        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = \
            m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)

        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration

        # 3) 写入 schedule
        self.schedule.add_split_or_merge(
            split_start, split_end, [ion],
            src_trap.id, dest_seg.id,
            Schedule.Split,
            split_swap_count, split_swap_hops, i1, i2, ion_swap_hops
        )

        self._annotate_last_event_with_shuttle_id()

        # 4) trace
        self._trace_add_step(
            "SPLIT", split_start, split_end,
            f"ion {ion}: T{src_trap.id} -> Seg{dest_seg.id} "
            f"(swap_cnt={split_swap_count}, swap_hops={split_swap_hops}, ion_hops={ion_swap_hops}, i1={i1}, i2={i2})"
        )
        self._trace_print(f"[TRACE] SPLIT ion={ion} T{src_trap.id} -> Seg{dest_seg.id} ({split_start}->{split_end})")

        return split_end

    def add_merge_op(self, clk, dest_trap, src_seg, ion):
        """
        把 ion 从 src_seg merge 回 dest_trap。
        """
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
        segment->segment 的 move（通过某个 junction）。
        论文贴合修复：
          junction 的阻塞由 junction_traffic_crossing 处理；
          不额外叠加 junction_cross_time，避免重复记时。
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

        # junction 交通冲突（同一 junction 同时过车），这是调度冲突约束，不是物理额外时间
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
        """
        在 trap_id 上执行 2Q gate。
        """
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
    # 小规模严格版目标选择：不使用 look-ahead
    # ==========================================================
    def _current_move_cost(self, source_trap, dest_trap):
        """
        小规模严格版只使用当前搬运代价，不看未来门。

        这里直接复用机器预计算的 trap-to-trap 距离；
        若缓存里没有，则退化为一个较大的默认值。
        """
        if source_trap == dest_trap:
            return 0
        return self.machine.dist_cache.get((source_trap, dest_trap), 100)

    # ==========================================================
    # Shuttle：Split / Move / Merge 链（论文意义的“跨区搬运”）
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        执行一次跨区搬运（论文意义 shuttle）：
          1) route 为空则使用 capacity-aware route
          2) 根据路径估计时间 -> identify_start_time
          3) 插入 split/move/merge
          4) sys_state 更新 trap 的离子顺序（保持原逻辑）
        """
        m = self.machine

        if len(route):
            rpath = route
        else:
            rpath = self._find_route_or_none(src_trap, dest_trap)
            if rpath is None:
                raise RuntimeError(f"No legal route found for shuttle: T{src_trap} -> T{dest_trap}")

        # ============ shuttle 计数 + trace record ============
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

        # --------- 估算总搬运用时（仅用于找 earliest feasible start）---------
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
        """
        保留原逻辑：
          - 找出路径中的 Trap 位置
          - 每段 Trap->...->Trap 做 partial_shuttle
          - 更新 sys_state：从源 trap 删除 ion，并按 orientation 插入目标 trap（保持链方向一致）
        """
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
        """
        partial path 必须是 Trap ... Trap（中间是 Junctions）
        """
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
    # MUSS Strict：冲突处理（rebalance）接口占位
    # ==========================================================
    def rebalance_traps(self, focus_traps, fire_time):
        """
        V6 中不再使用全局 rebalance。

        保留该接口仅为了兼容旧调用链与外部统计脚本。
        真正的论文式 conflict handling 已下沉到：
          _prepare_meeting_trap() / _ensure_space_on_trap()
        也就是“先选目标执行 zone，再做局部 LRU 驱逐”。
        """
        return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        """
        兼容占位接口。V6 不走全局 rebalance 路径。
        """
        return fire_time

    def _gate_payload(self, gate):
        data = self.gate_info.get(gate, None)
        if data is None:
            return None, [], "unknown"
        if isinstance(data, dict):
            return data, list(data.get("qubits", [])), data.get("type", "unknown")
        return {"qubits": list(data), "type": "cx" if len(data) == 2 else "u"}, list(data), "cx" if len(data) == 2 else "u"

    def _candidate_meeting_traps(self, trap_a, trap_b):
        """
        Small mode:
            no partition semantics, all traps can execute local 2Q gates.
            Current stable implementation still meets at one endpoint trap only.

        Large mode:
            preserve endpoint-only meeting for the current local shuttle path,
            but order the two endpoints by zone preference:
                optical > operation > storage
        """
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
        Table 2 小规模严格版：
        只允许在 ion1_trap / ion2_trap 之一执行 2Q gate。

        与前一版不同，这里不再使用 future-score / look-ahead，
        只按“当前谁搬过去代价更小”来做 endpoint 目标选择。

        规则：
          1) 候选 target trap 必须当前还能再接收 1 个离子
          2) 若两个 endpoint 都可作为目标，则比较当前搬运代价
          3) 若当前搬运代价相同，则按 _candidate_meeting_traps 给出的顺序做稳定 tie-break

        返回：
          (moving_ion, target_trap) 或 None
        """
        candidates = self._candidate_meeting_traps(ion1_trap, ion2_trap)
        best = None
        best_score = None

        for target in candidates:
            if not self._trap_has_free_slot(target, incoming=1):
                continue

            if target == ion1_trap:
                moving = ion2
                move_cost = self._current_move_cost(ion2_trap, target)
            elif target == ion2_trap:
                moving = ion1
                move_cost = self._current_move_cost(ion1_trap, target)
            else:
                continue

            score = move_cost

            if best_score is None or score < best_score:
                best = (moving, target)
                best_score = score

        return best

    def _trap_zone_type(self, trap_id):
        if not hasattr(self.machine, "get_trap"):
            return "storage"
        try:
            return getattr(self.machine.get_trap(trap_id), "zone_type", "storage")
        except Exception:
            return "storage"

    def _zone_level(self, zone_type):
        """
        论文中的 multi-level 语义：
          storage = level 0
          operation = level 1
          optical = level 2
        未知类型默认按 storage 处理。
        """
        return {"storage": 0, "operation": 1, "optical": 2}.get(str(zone_type), 0)

    def _trap_level(self, trap_id):
        return self._zone_level(self._trap_zone_type(trap_id))

    def _candidate_traps_for_eviction(self, src_trap):
        """
        论文的 conflict handling：
        当目标 zone 满时，把其中一个“长期未使用”的离子迁移到其它更合适的 zone。

        这里的排序原则是：
          1) 优先迁往不高于当前 trap level 的 zone（符合“从高层逐步回落到低层”的论文描述）
          2) 在 level 合法时优先 level 更近
          3) 再优先图距离更近
          4) 最后按 trap id 稳定打破平局
        """
        src_level = self._trap_level(src_trap)
        cand = []
        for trap in self.machine.traps:
            tid = trap.id
            if tid == src_trap:
                continue
            if not self._trap_has_free_slot(tid, incoming=1):
                continue
            dst_level = self._trap_level(tid)
            downward_penalty = 0 if dst_level <= src_level else 1000
            level_gap = abs(src_level - dst_level)
            graph_dist = self.machine.dist_cache.get((src_trap, tid), 10 ** 6)
            cand.append((downward_penalty, level_gap, graph_dist, tid))
        cand.sort()
        return [x[-1] for x in cand]

    def _select_lru_victim(self, trap_id, forbidden_ions=None):
        """
        从 trap 中选择一个可驱逐离子：
          - 不允许驱逐当前 gate 涉及离子
          - 使用论文明确提到的 LRU 策略
        """
        if forbidden_ions is None:
            forbidden_ions = set()
        ions = list(self.sys_state.trap_ions[trap_id])
        candidates = [ion for ion in ions if ion not in forbidden_ions]
        if not candidates:
            return None
        return min(candidates, key=lambda ion: self.ion_last_used.get(ion, -1))

    def _ensure_space_on_trap(self, trap_id, fire_time, forbidden_ions=None):
        """
        局部冲突处理核心：
        若 trap 已满，则按照论文路径执行：
          1) 在该 trap 中选 LRU victim
          2) 按 multi-level scheduling 选择新的目标 trap
          3) 做一次合法 shuttle，把 victim 移走
          4) 直到 trap 出现空位或确认无解

        返回：
          (success, new_time)
        """
        if forbidden_ions is None:
            forbidden_ions = set()

        cur_time = fire_time
        guard = 0
        while not self._trap_has_free_slot(trap_id, incoming=1):
            guard += 1
            if guard > len(self.machine.traps) + 8:
                return False, cur_time

            victim = self._select_lru_victim(trap_id, forbidden_ions=forbidden_ions)
            if victim is None:
                return False, cur_time

            moved = False
            victim_ready, victim_trap = self.ion_ready_info(victim)
            cur_time = max(cur_time, victim_ready)

            for dst_trap in self._candidate_traps_for_eviction(victim_trap):
                route = self._find_route_or_none(victim_trap, dst_trap)
                if route is None:
                    continue
                cur_time = self.fire_shuttle(victim_trap, dst_trap, victim, cur_time, route=route)
                moved = True
                break

            if not moved:
                return False, cur_time

        return True, cur_time

    def _choose_preferred_target_ignore_capacity(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        当两个 endpoint trap 当前都满时，论文 3.2 的小规模退化口径不是立刻死锁，
        而是：先在两个 endpoint 中选一个更适合执行当前 gate 的目标 trap，
        然后再只围绕这个目标 trap 做 conflict handling。

        本版仍然不使用 look-ahead，只按当前搬运代价决定偏好。
        返回：
          (moving_ion, target_trap, score) 或 None
        """
        candidates = self._candidate_meeting_traps(ion1_trap, ion2_trap)
        best = None
        for target in candidates:
            if target == ion1_trap:
                moving = ion2
                move_cost = self._current_move_cost(ion2_trap, target)
            elif target == ion2_trap:
                moving = ion1
                move_cost = self._current_move_cost(ion1_trap, target)
            else:
                continue

            score = move_cost
            item = (moving, target, score)
            if best is None or score < best[2]:
                best = item
        return best

    def _prepare_meeting_trap(self, ion1_trap, ion2_trap, ion1, ion2, fire_time, gate_idx):
        """
        为当前 2Q gate 选择执行 trap，并在必要时做论文式 conflict handling。

        返回：
          (success, moving_ion, source_trap, dest_trap, ready_time)
        """
        meet_choice = self._choose_partition_target(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
        cur_time = fire_time

        if meet_choice is not None:
            moving_ion, dest_trap = meet_choice
            source_trap = ion1_trap if moving_ion == ion1 else ion2_trap
            return True, moving_ion, source_trap, dest_trap, cur_time

        preferred = self._choose_preferred_target_ignore_capacity(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
        if preferred is None:
            return False, None, None, None, cur_time

        moving_ion, dest_trap, _ = preferred
        source_trap = ion1_trap if moving_ion == ion1 else ion2_trap

        ok, cur_time = self._ensure_space_on_trap(
            dest_trap,
            cur_time,
            forbidden_ions={ion1, ion2},
        )
        if not ok:
            return False, None, None, None, cur_time

        return True, moving_ion, source_trap, dest_trap, cur_time
        return best

    # ==========================================================
    # Gate scheduling：按 frontier 逐个 gate 执行（MUSS strict）
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        Paper-faithful MUSS gate scheduling:
          - 1Q gates do NOT participate in the MUSS frontier loop.
          - This method is for 2Q gates only.
          - 1Q gates are inserted later in _schedule_delayed_one_qubit_gates().
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

        if self.is_large_mode:
            t1_obj = self.machine.get_trap(ion1_trap)
            t2_obj = self.machine.get_trap(ion2_trap)
            if getattr(t1_obj, "qccd_id", 0) != getattr(t2_obj, "qccd_id", 0):
                raise NotImplementedError(
                    "Large-scale cross-QCCD optical/fiber + inter-module swap-insert path "
                    "is explicitly reserved for the next stage and is not enabled in this file yet."
                )

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
            # V6.2-small：严格按小规模 3.2 口径推进：先选 gate / 再选目标 trap / 再做局部 conflict handling / 最后执行 gate。
            ok, moving_ion, source_trap, dest_trap, prep_time = self._prepare_meeting_trap(
                ion1_trap, ion2_trap, ion1, ion2, fire_time, gate_idx
            )
            if not ok:
                self.protected_ions = set()
                raise RuntimeError(
                    "Scheduling deadlock under paper-faithful local conflict handling. "
                    f"gate={gate}, ion1={ion1}, ion2={ion2}, "
                    f"ion1_trap={ion1_trap}, ion2_trap={ion2_trap}, fire_time={fire_time}"
                )

            route = self._find_route_or_none(source_trap, dest_trap)
            if route is None:
                # 若此时仍无路，说明并非简单的目标 trap 满，而是整个局部区域都被堵住。
                # 继续尝试对目标 trap 做一次额外局部腾挪；若仍失败，再明确报错。
                ok2, prep_time = self._ensure_space_on_trap(
                    dest_trap, prep_time, forbidden_ions={ion1, ion2}
                )
                if ok2:
                    route = self._find_route_or_none(source_trap, dest_trap)

            if route is None:
                self.protected_ions = set()
                raise RuntimeError(
                    "Scheduling deadlock: no legal capacity-aware route exists after local conflict handling. "
                    f"gate={gate}, moving_ion={moving_ion}, source_trap={source_trap}, "
                    f"dest_trap={dest_trap}, fire_time={prep_time}"
                )

            clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, prep_time, route=route)

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

        self.ion_last_used[ion1] = finish_time
        self.ion_last_used[ion2] = finish_time
        self.protected_ions = set()

    def add_one_qubit_gate(self, gate):
        """
        Insert a 1Q gate after all 2Q scheduling has established ion trajectories.
        1Q gates affect final time/fidelity, but never participate in the MUSS frontier.
        """
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

    def is_executable_local(self, gate):
        """2Q helper: whether the gate can execute locally without a shuttle."""
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) != 2:
            return False
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # 主循环：Frontier scheduling（MUSS strict）
    # ==========================================================
    def run(self):
        """
        Paper-faithful MUSS frontier:
          1) Run MUSS only on the 2Q-only DAG.
          2) 1Q gates do not participate in gate selection.
          3) After all 2Q gates are scheduled, replay the full DAG and insert 1Q gates
             at their earliest legal times so they still affect total time/fidelity.
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

            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)

        self._schedule_delayed_one_qubit_gates()
