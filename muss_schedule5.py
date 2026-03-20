# muss_schedule2.py
# ============================================================
# MUSS Scheduler (V6) —— 论文严格口径版（局部 LRU 驱逐，去掉全局 rebalance）
#
# 设计原则：
# 1) 保留原有外部接口和主调度框架，不随意修改下游依赖。
# 2) 保留：
#    - frontier 只在 2Q DAG 上调度
#    - 1Q gate 延后插入
#    - ion_ready_info() 强一致性检查
#    - FreeTrapRoute 作为合法路径检查/执行层
#    - fire_shuttle() 生成 split / move / merge 事件链
#    - shuttle_id 标注
# 3) 去掉原始项目的全局 rebalance.py。
# 4) 当当前 gate 因为目标 trap 满、两端都堵、或 route 不可达而无法推进时，
#    仅执行论文语义内的“局部冲突处理”：
#       - LRU 选择被驱逐 ion
#       - 保护当前 gate 的两个 ion 不被驱逐
#       - 优先驱逐到更低层 / 更外围 / 更不关键的 trap
#       - 每一步驱逐仍必须经过 FreeTrapRoute 验证合法路径
# 5) 不引入新的全局优化器，不做全局 flow，不做全局 trap clearing。
#
# 说明：
# - 这版的“rebalance_traps / do_rebalance_traps”名字仍保留，
#   只是为了兼容原代码内部调用结构；其实现已经不再是全局 rebalance，
#   而是局部 LRU 冲突消解。
# - Small-scale（你当前 Table 2 复现路径）通常没有显式 partition level；
#   这时“lower-level zone”的语义会退化为：
#       最近、合法、非当前 focus trap、且有空位的 trap
#   这是为了在不改机器接口的前提下，保持论文语义尽可能接近。
# ============================================================

#比“最严格的论文口径”还是略微更激进，因为它会在 route 不通时对 focus traps 做额外局部疏通；这一步从工程上合理，但比“只对选定目标 zone 做 conflict handling”#更宽一点


import networkx as nx
import numpy as np
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction


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
        New paper-faithful interface:
            MUSSSchedule(parse_obj, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)

        Backward-compatible legacy interface:
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

        # 保留原结构：BasicRoute 仅作调试兜底；主路径仍使用 FreeTrapRoute
        self.basic_router = BasicRoute(self.machine)
        self.router = None
        self.gate_finish_times = {}

        # 调度统计
        self.count_rebalance = 0  # V6 中记录“局部冲突消解”次数
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

        # 预计算 trap-to-trap 距离（供 lookahead / 目标打分使用）
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === MUSS Strict Requirement: LRU Tracking ===
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # 当前 gate 的 qubit 不允许被驱逐
        self.protected_ions = set()

        # 记录最近一次需要做局部冲突消解的 focus traps（兼容 do_rebalance_traps 调用）
        self._last_focus_traps = None

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
    # Trap / zone 元数据辅助
    # ==========================================================
    def _get_trap_obj(self, trap_id):
        try:
            return self.machine.get_trap(trap_id)
        except Exception:
            return self.machine.traps[trap_id]

    def _trap_qccd_id(self, trap_id):
        try:
            return getattr(self._get_trap_obj(trap_id), "qccd_id", 0)
        except Exception:
            return 0

    def _zone_type(self, trap_id):
        try:
            return getattr(self._get_trap_obj(trap_id), "zone_type", "storage")
        except Exception:
            return "storage"

    def _zone_priority(self, trap_id):
        """
        数值越小表示越“高层/关键”：
            optical < operation < storage
        驱逐时优先往“更低层”（即数值更大）的 trap 走。
        """
        order = {"optical": 0, "operation": 1, "storage": 2}
        return order.get(self._zone_type(trap_id), 2)

    def _zone_level(self, trap_id):
        """
        若机器显式提供 zone_level，则使用之；
        否则退化为 zone_priority。
        """
        try:
            obj = self._get_trap_obj(trap_id)
            if hasattr(obj, "zone_level"):
                return int(getattr(obj, "zone_level"))
        except Exception:
            pass
        return self._zone_priority(trap_id)

    def _same_qccd(self, trap_a, trap_b):
        return self._trap_qccd_id(trap_a) == self._trap_qccd_id(trap_b)

    # ==========================================================
    # 容量 / 路由辅助
    # ==========================================================
    def _trap_has_free_slot(self, trap_id, incoming=1):
        cur = len(self.sys_state.trap_ions[trap_id])
        cap = self.machine.traps[trap_id].capacity
        return (cur + incoming) <= cap

    def _trap_free_slots(self, trap_id):
        return self.machine.traps[trap_id].capacity - len(self.sys_state.trap_ions[trap_id])

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
    # Lookahead：决定“谁去找谁”（论文 strict + 未来门预测）
    # ==========================================================
    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        """
        未来代价预测：看接下来若干个 gate 中 ion 的伙伴在哪，
        用 trap-to-trap hop 距离做一个折扣累计。
        """
        score = 0
        gamma = 1.0
        weight = 1.0
        lookahead_depth = 8
        found_count = 0

        total_gates = len(self.gates)
        start_idx = current_gate_idx + 1 if current_gate_idx is not None else 0

        for i in range(start_idx, total_gates):
            if found_count >= lookahead_depth:
                break

            next_gate = self.gates[i]
            if next_gate in self.gate_info:
                gate_data = self.gate_info[next_gate]
                q_list = gate_data if isinstance(gate_data, list) else gate_data["qubits"]

                if ion in q_list:
                    # 1Q gate 不产生跨阱需求
                    if len(q_list) == 1:
                        continue

                    partner = q_list[1] if q_list[0] == ion else q_list[0]
                    _, partner_trap = self.ion_ready_info(partner)

                    dist = self.machine.dist_cache.get((target_trap_id, partner_trap), 10)
                    score += weight * dist
                    weight *= gamma
                    found_count += 1

        return score

    def shuttling_direction(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        决定哪个离子搬到对方阱：
        同时考虑：
          - 当前搬运代价（trap-hop 距离）
          - 未来代价（lookahead）
          - capacity 约束
        返回:
          (source_trap, dest_trap)
        若两个方向都非法，返回 (None, None)
        """
        m = self.machine
        ALPHA = 1

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

        # Option A: ion1 去 ion2（在 Trap2 相遇）
        cost_current_move1 = m.dist_cache.get((ion1_trap, ion2_trap), 100)
        future_score_1 = self.get_future_score(ion1, current_gate_idx, ion2_trap)
        future_score_2 = self.get_future_score(ion2, current_gate_idx, ion2_trap)
        total_score_meet_at_t2 = cost_current_move1 + ALPHA * (future_score_1 + future_score_2)

        # Option B: ion2 去 ion1（在 Trap1 相遇）
        cost_current_move2 = m.dist_cache.get((ion2_trap, ion1_trap), 100)
        future_score_1_at_t1 = self.get_future_score(ion1, current_gate_idx, ion1_trap)
        future_score_2_at_t1 = self.get_future_score(ion2, current_gate_idx, ion1_trap)
        total_score_meet_at_t1 = cost_current_move2 + ALPHA * (future_score_1_at_t1 + future_score_2_at_t1)

        if total_score_meet_at_t2 < total_score_meet_at_t1:
            return ion1_trap, ion2_trap
        else:
            return ion2_trap, ion1_trap

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
    # V6 核心：局部 LRU 冲突处理（替代全局 rebalance）
    # ==========================================================
    def _pick_lru_victim(self, trap_id):
        """
        从 trap_id 中选一个可驱逐 ion：
          - 当前 gate 的 qubit（protected_ions）绝不驱逐
          - 其余 ion 按 LRU（最久未使用）选
        """
        candidates = self.sys_state.trap_ions[trap_id]
        valid_candidates = [ion for ion in candidates if ion not in self.protected_ions]

        if not valid_candidates:
            return None

        return min(valid_candidates, key=lambda ion: self.ion_last_used.get(ion, -1))

    def _candidate_evict_destinations(self, source_trap, avoid_traps=None):
        """
        为 source_trap 选择局部驱逐目的地。
        论文语义希望“往更低层 zone 驱逐”，这里按以下顺序选：

        1) 同一 QCCD 内、且 level 更低（或 zone_priority 更低层）的 trap
        2) 同一 QCCD 内任意有空位 trap
        3) 小规模无显式分层时：全机器最近合法空闲 trap

        同时要求：
        - 目标 trap 有空位
        - 不在 avoid_traps 中
        - source -> dest 存在合法 FreeTrapRoute
        """
        if avoid_traps is None:
            avoid_traps = set()
        else:
            avoid_traps = set(avoid_traps)

        src_level = self._zone_level(source_trap)
        src_qccd = self._trap_qccd_id(source_trap)

        strict_same_qccd_lower = []
        relaxed_same_qccd = []
        global_fallback = []

        for tr in self.machine.traps:
            tid = tr.id
            if tid == source_trap:
                continue
            if tid in avoid_traps:
                continue
            if not self._trap_has_free_slot(tid, incoming=1):
                continue

            route = self._find_route_or_none(source_trap, tid)
            if route is None:
                continue

            dst_level = self._zone_level(tid)
            same_qccd = (self._trap_qccd_id(tid) == src_qccd)

            item = (
                self.machine.trap_distance(source_trap, tid),   # 先看距离
                -self._trap_free_slots(tid),                    # 再看空位多寡（空位多更好）
                tid,
                route,
            )

            if same_qccd and dst_level > src_level:
                strict_same_qccd_lower.append(item)
            elif same_qccd:
                relaxed_same_qccd.append(item)
            else:
                global_fallback.append(item)

        strict_same_qccd_lower.sort()
        relaxed_same_qccd.sort()
        global_fallback.sort()

        ordered = strict_same_qccd_lower + relaxed_same_qccd + global_fallback
        return ordered

    def _local_evict_one(self, source_trap, fire_time, avoid_traps=None):
        """
        在 source_trap 做一次“局部 LRU 驱逐”：
          - 选 victim（非 protected 且最久未使用）
          - 选合法 destination
          - 执行一次 fire_shuttle
        返回:
          (是否成功驱逐, 新完成时间)
        """
        victim = self._pick_lru_victim(source_trap)
        if victim is None:
            return False, fire_time

        destinations = self._candidate_evict_destinations(source_trap, avoid_traps=avoid_traps)
        if not destinations:
            return False, fire_time

        ion_time, _ = self.ion_ready_info(victim)
        start_time = max(fire_time, ion_time)

        for _, _, dst_trap, route in destinations:
            try:
                fin_time = self.fire_shuttle(source_trap, dst_trap, victim, start_time, route=route)
                return True, fin_time
            except Exception:
                # 当前目标失败则尝试下一个候选
                continue

        return False, fire_time

    def _local_conflict_relief(self, focus_traps, fire_time):
        """
        对当前 gate 附近的冲突做“局部”处理，而不是全局 rebalance。

        触发条件（与原来 rebalance_traps 的判定口径保持接近）：
          - 两个 focus trap 都满
          - 或者两个方向都 route 不通
          - 或者当前候选目标 trap 没有空位 / 无合法路

        处理方式：
          - 优先从已满的 focus trap 驱逐
          - 若容量不满但仍 route 双向不通，则允许从 focus trap 做一次局部疏通
          - 一次调用最多做有限步局部驱逐，避免死循环

        返回:
          (是否发生了局部疏通, 最新完成时间)
        """
        self.count_rebalance += 1

        if len(focus_traps) < 2:
            focus_traps = list(focus_traps) + list(focus_traps)

        t1 = focus_traps[0]
        t2 = focus_traps[1]
        fin_time = fire_time
        progress = False

        max_rounds = max(2, len(self.sys_state.trap_ions))  # 保守上界，防止无限循环

        for _ in range(max_rounds):
            excess_cap1 = self.machine.traps[t1].capacity - len(self.sys_state.trap_ions[t1])
            excess_cap2 = self.machine.traps[t2].capacity - len(self.sys_state.trap_ions[t2])

            status12, _ = self.router.find_route(t1, t2)
            status21, _ = self.router.find_route(t2, t1)

            both_full = (excess_cap1 == 0 and excess_cap2 == 0)
            both_blocked = (status12 == 1 and status21 == 1)

            if not both_full and not both_blocked:
                break

            round_progress = False

            # 优先释放已满 trap 的空间
            candidate_sources = []
            if excess_cap1 == 0:
                candidate_sources.append(t1)
            if excess_cap2 == 0 and t2 not in candidate_sources:
                candidate_sources.append(t2)

            # 如果容量上看没满，但双向 route 都堵，则对两个 focus trap 都尝试做一次局部疏通
            if not candidate_sources:
                candidate_sources = [t1]
                if t2 != t1:
                    candidate_sources.append(t2)

            for src in candidate_sources:
                changed, fin_time = self._local_evict_one(
                    src,
                    fin_time,
                    avoid_traps=set([t1, t2])  # 先不要把被驱逐 ion 又塞进另一个 focus trap
                )
                if changed:
                    round_progress = True
                    progress = True
                    # 做一次就回到 while 顶部重新看可达性，避免过度搬运
                    break

            if not round_progress:
                break

        return progress, fin_time

    # ==========================================================
    # MUSS Strict：兼容旧名字，但实现为“局部 LRU 冲突处理”
    # ==========================================================
    def rebalance_traps(self, focus_traps, fire_time):
        """
        兼容旧调用结构：
        原先这里会调用全局 rebalance；
        V6 中这里改成“只对当前 gate 附近做局部 LRU 疏通”。

        返回值保持不变：
          (1, fin_time) -> 发生了局部疏通，调用者应重试 schedule_gate
          (0, fire_time) -> 没有做任何局部疏通
        """
        self._last_focus_traps = tuple(focus_traps)

        changed, fin_time = self._local_conflict_relief(focus_traps, fire_time)
        if changed:
            return 1, fin_time
        return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        """
        兼容旧接口名。
        在 V6 中，不再做全局 rebalance，而是对最近一次记录的 focus_traps
        做局部 LRU 冲突消解。

        如果最近没有记录 focus_traps，则直接返回原时间。
        """
        if self._last_focus_traps is None:
            return fire_time

        changed, fin_time = self._local_conflict_relief(self._last_focus_traps, fire_time)
        if changed:
            return fin_time

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
            当前稳定实现仍保持“只在两个 endpoint trap 中选择 meeting trap”，
            以保证不偏离你当前 V2/V5 的接口和统计口径。

        Large mode:
            仍保持 endpoint-only，但按 zone preference 排序：
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
        Table 2 stable mode：
        只允许在 ion1_trap / ion2_trap 之一执行 2Q gate。
        保留原修复：
          - 候选 target trap 必须有空位
          - 若两个 target 都非法，则返回 None
        返回值:
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

    def _try_route_after_local_relief(self, source_trap, dest_trap, fire_time):
        """
        当 source->dest 当前没有合法 route 时，尝试对局部做一次疏通后重试。
        返回:
          (route_or_none, new_fire_time)
        """
        route = self._find_route_or_none(source_trap, dest_trap)
        if route is not None:
            return route, fire_time

        changed, fin_time = self._local_conflict_relief([source_trap, dest_trap], fire_time)
        if not changed:
            return None, fire_time

        route = self._find_route_or_none(source_trap, dest_trap)
        return route, fin_time

    # ==========================================================
    # Gate scheduling：按 frontier 逐个 gate 执行（MUSS strict）
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        Paper-faithful MUSS gate scheduling:
          - 1Q gates do NOT participate in the MUSS frontier loop.
          - This method is for 2Q gates only.
          - 1Q gates are inserted later in _schedule_delayed_one_qubit_gates().

        V6 变化点仅在“无路可走时的冲突处理”：
          - 不再调用全局 rebalance.py
          - 改为局部 LRU 驱逐 + 合法路径重试
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

        # ------------------------------------------------------
        # Case 1: 两个 ion 已经同 trap，直接发 gate
        # ------------------------------------------------------
        if ion1_trap == ion2_trap:
            gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
            zone_type = getattr(self.machine.get_trap(ion1_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(
                fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap,
                gate_type=gate_type, zone_type=zone_type, gate_id=gate
            )
            self.gate_finish_times[gate] = fire_time + gate_duration
            finish_time = fire_time + gate_duration

        # ------------------------------------------------------
        # Case 2: 需要 shuttle 后再 gate
        # ------------------------------------------------------
        else:
            # Step A: 若当前两个 focus trap 明显僵住，则先做一次局部冲突消解，然后整体重试本 gate
            rebal_flag, new_fin_time = self.rebalance_traps(
                focus_traps=[ion1_trap, ion2_trap],
                fire_time=fire_time
            )
            if rebal_flag:
                self.protected_ions = set()
                self.schedule_gate(gate, specified_time=new_fin_time, gate_idx=gate_idx)
                return

            # Step B: 先按原逻辑，优先在两个 endpoint 中选目标 trap
            meet_choice = self._choose_partition_target(ion1_trap, ion2_trap, ion1, ion2, gate_idx)

            if meet_choice is not None:
                moving_ion, dest_trap = meet_choice
                source_trap = ion1_trap if moving_ion == ion1 else ion2_trap

                # 若目标 trap 当前没空位（理论上 _choose_partition_target 已过滤，但局部状态可能变化），
                # 则对目标局部做一次疏通
                if not self._trap_has_free_slot(dest_trap, incoming=1):
                    changed, new_fire = self._local_conflict_relief([source_trap, dest_trap], fire_time)
                    if changed:
                        self.protected_ions = set()
                        self.schedule_gate(gate, specified_time=new_fire, gate_idx=gate_idx)
                        return

            else:
                # Step C: endpoint 目标都不合法，则回落到原本的 bidirectional direction 选择
                source_trap, dest_trap = self.shuttling_direction(
                    ion1_trap, ion2_trap, ion1, ion2, gate_idx
                )

                if source_trap is None or dest_trap is None:
                    # 两个方向都不合法：尝试局部冲突消解后整体重试
                    new_fire = self.do_rebalance_traps(fire_time)
                    if new_fire > fire_time:
                        self.protected_ions = set()
                        self.schedule_gate(gate, specified_time=new_fire, gate_idx=gate_idx)
                        return

                    self.protected_ions = set()
                    raise RuntimeError(
                        "Scheduling dead-end after local LRU conflict handling: both meeting directions are illegal. "
                        f"gate={gate}, ion1={ion1}, ion2={ion2}, "
                        f"ion1_trap={ion1_trap}, ion2_trap={ion2_trap}, fire_time={fire_time}"
                    )

                moving_ion = ion1 if source_trap == ion1_trap else ion2

            # Step D: 对当前确定的 source/dest，若无合法 route，则局部疏通后重试 route
            route, route_ready_time = self._try_route_after_local_relief(source_trap, dest_trap, fire_time)
            if route is None:
                self.protected_ions = set()
                raise RuntimeError(
                    "Scheduling dead-end after local LRU conflict handling: no legal capacity-aware route exists. "
                    f"gate={gate}, moving_ion={moving_ion}, source_trap={source_trap}, "
                    f"dest_trap={dest_trap}, fire_time={fire_time}"
                )

            # Step E: 合法 shuttle
            clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, route_ready_time, route=route)

            # Step F: shuttle 后必须验证两个 ion 已经共址
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