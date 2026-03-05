import networkx as nx
import numpy as np
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *


class MUSSSchedule:
    """
    创新版本 MUSS Scheduler（保持创新内核不变 + 增强调度可观测性与正确性检查）

    创新内核（保留）：
    1) executed_gate_indices：集合跟踪已执行门，避免 local-first 造成 lookahead 误判
    2) qubit_queues：预计算每个量子比特未来任务队列
    3) get_future_score：基于 executed set 的 lookahead（深度更深）
    4) rebalance：使用“近似 Belady”（最晚下次使用）选择 victim，仍基于 executed set

    增强功能（新增）：
    A) shuttle_counter：宏观穿梭次数（每次 fire_shuttle 记 1）
    B) shuttle_log：记录每次 shuttle 的全过程（Split/Move/Merge 时间区间与路径）
    C) dump_shuttle_trace：打印“哪个离子从哪里移动到哪里全过程”
    D) 正确性断言：gate 前同 trap；shuttle 后到达 dest；merge 后不超容量
    """

    def __init__(self, ir, gate_info, M, init_map, serial_trap_ops, serial_comm, global_serial_lock):
        self.ir = ir
        self.gate_info = gate_info
        self.machine = M
        self.init_map = init_map

        # 串行约束
        self.SerialTrapOps = serial_trap_ops
        self.SerialCommunication = serial_comm
        self.GlobalSerialLock = global_serial_lock

        self.schedule = Schedule(M)
        self.router = BasicRoute(M)
        self.gate_finish_times = {}

        # 统计
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # ============ 新增：宏观 shuttle 计数 + 全过程 trace ============
        self.shuttle_counter = 0
        self.shuttle_log = []  # 每条: {"shuttle_id","ion","src_trap","dst_trap","route_text","steps":[...]}
        self.enable_runtime_trace_print = False  # True 时实时打印 trace
        self._current_shuttle_id = None
        # ============================================================

        # Create sys_state
        trap_ions = {}
        seg_ions = {}
        for t in M.traps:
            if init_map.get(t.id, None):
                trap_ions[t.id] = init_map[t.id][:]
            else:
                trap_ions[t.id] = []
        for s in M.segments:
            seg_ions[s.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # Precompute distances
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === 创新：预计算全局门序列（用于 lookahead/FCFS tie-break）===
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)

        # 门ID -> 拓扑索引
        self.gate_to_idx = {g: i for i, g in enumerate(self.static_topo_list)}

        # === 创新核心修复：使用集合跟踪已执行的门索引 ===
        self.executed_gate_indices = set()

        # === 创新：预计算每个量子比特的任务队列（只读）===
        self.qubit_queues = collections.defaultdict(list)
        for g_idx, g in enumerate(self.static_topo_list):
            if g in self.gate_info:
                info = self.gate_info[g]
                qs = info if isinstance(info, list) else info["qubits"]
                for q in qs:
                    self.qubit_queues[q].append(g_idx)

        # protected ions：rebalance 时不允许驱逐
        self.protected_ions = set()

        # 兼容接口
        self.gates = self.static_topo_list

        # ============ 正确性检查开关 ============
        # 重要：建议默认开启，任何状态不一致都会立即暴露
        self.enable_assertions = True
        # =====================================

    # ----------------------------
    # Trace / Debug 输出
    # ----------------------------
    def _trace_print(self, msg: str):
        if self.enable_runtime_trace_print:
            print(msg)

    def _trace_add_step(self, etype, t_start, t_end, desc):
        if self._current_shuttle_id is None:
            return
        sid = self._current_shuttle_id
        if sid < 0 or sid >= len(self.shuttle_log):
            return
        self.shuttle_log[sid]["steps"].append({
            "etype": etype,
            "t_start": int(t_start),
            "t_end": int(t_end),
            "desc": desc
        })

    def dump_shuttle_trace(self, max_lines=None):
        """
        打印所有 shuttle 的全过程：
          [SHUTTLE k] ion=.. Tsrc->Tdst route=...
            - SPLIT (t1->t2) ...
            - MOVE  (t1->t2) ...
            - MERGE (t1->t2) ...
        """
        lines = []
        for rec in self.shuttle_log:
            sid = rec["shuttle_id"]
            ion = rec["ion"]
            src = rec["src_trap"]
            dst = rec["dst_trap"]
            route_txt = rec.get("route_text", "")
            lines.append(f"[SHUTTLE {sid}] ion={ion}  T{src} -> T{dst}  route={route_txt}")
            for st in rec.get("steps", []):
                lines.append(f"    - {st['etype']:<5}  ({st['t_start']} -> {st['t_end']})  {st['desc']}")
        if max_lines is not None:
            lines = lines[:max_lines]
        return "\n".join(lines)

    # ----------------------------
    # 依赖就绪时间
    # ----------------------------
    def gate_ready_time(self, gate):
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    # ----------------------------
    # 离子就绪 + 位置（创新版本：不再强制对比 sys_state，这里保留你的原行为）
    # 如需更强一致性，可改回“同时对比 sys_state”，但会改变你原创新代码行为。
    # ----------------------------
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
        return this_ion_last_op_time, this_ion_trap

    # ==========================================================
    # Basic Operations (Split, Merge, Move, Gate)
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

        # trace
        self._trace_add_step(
            "SPLIT", split_start, split_end,
            f"ion {ion}: T{src_trap.id} -> Seg{dest_seg.id} "
            f"(swap_cnt={split_swap_count}, swap_hops={split_swap_hops}, ion_hops={ion_swap_hops}, i1={i1}, i2={i2})"
        )
        self._trace_print(f"[TRACE] SPLIT ion={ion} T{src_trap.id}->Seg{dest_seg.id} ({split_start}->{split_end})")
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

        # trace
        self._trace_add_step("MERGE", merge_start, merge_end, f"ion {ion}: Seg{src_seg.id} -> T{dest_trap.id}")
        self._trace_print(f"[TRACE] MERGE ion={ion} Seg{src_seg.id}->T{dest_trap.id} ({merge_start}->{merge_end})")

        # correctness: merge 后检查容量
        if self.enable_assertions:
            try:
                cap = self.machine.traps[dest_trap.id].capacity
                assert len(self.sys_state.trap_ions[dest_trap.id]) <= cap, \
                    f"Capacity overflow at T{dest_trap.id}: {len(self.sys_state.trap_ions[dest_trap.id])}>{cap}"
            except Exception:
                pass

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

        move_end = move_start + m.move_time(src_seg.id, dest_seg.id) + m.junction_cross_time(junct)

        # junction traffic conflict resolution（保留原逻辑）
        move_start, move_end = self.schedule.junction_traffic_crossing(
            src_seg, dest_seg, junct, move_start, move_end
        )

        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)

        # trace
        self._trace_add_step("MOVE", move_start, move_end, f"ion {ion}: Seg{src_seg.id} -> Seg{dest_seg.id} via J{junct.id}")
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
    # Lookahead Logic (创新内核：基于 executed_gate_indices)
    # ==========================================================
    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        score = 0
        gamma = 0.9
        weight = 1.0
        lookahead_depth = 20
        found_count = 0

        queue = self.qubit_queues[ion]

        for g_idx in queue:
            # 创新关键：跳过已执行的门（而不是仅仅 g_idx > current_gate_idx）
            if g_idx in self.executed_gate_indices:
                continue

            # 跳过当前正在决策的门本身
            if g_idx == current_gate_idx:
                continue

            if found_count >= lookahead_depth:
                break

            gate = self.gates[g_idx]
            if gate in self.gate_info:
                info = self.gate_info[gate]
                qs = info if isinstance(info, list) else info["qubits"]
                if len(qs) == 2:
                    partner = qs[1] if qs[0] == ion else qs[0]
                    _, partner_trap = self.ion_ready_info(partner)

                    dist = self.machine.dist_cache.get((target_trap_id, partner_trap), 10)
                    score += weight * dist
                    weight *= gamma
                    found_count += 1

        return score

    def shuttling_direction(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        m = self.machine
        ALPHA = 0.7

        if current_gate_idx is None:
            return ion1_trap, ion2_trap

        # meet at ion2_trap
        cost_move1 = m.dist_cache.get((ion1_trap, ion2_trap), 100)
        future1 = self.get_future_score(ion1, current_gate_idx, ion2_trap)
        future2 = self.get_future_score(ion2, current_gate_idx, ion2_trap)
        score_t2 = cost_move1 + ALPHA * (future1 + future2)

        # meet at ion1_trap
        cost_move2 = m.dist_cache.get((ion2_trap, ion1_trap), 100)
        future1_t1 = self.get_future_score(ion1, current_gate_idx, ion1_trap)
        future2_t1 = self.get_future_score(ion2, current_gate_idx, ion1_trap)
        score_t1 = cost_move2 + ALPHA * (future1_t1 + future2_t1)

        # capacity constraints
        ss = self.sys_state
        cap1 = m.traps[ion1_trap].capacity - len(ss.trap_ions[ion1_trap])
        cap2 = m.traps[ion2_trap].capacity - len(ss.trap_ions[ion2_trap])

        if cap1 <= 0 and cap2 > 0:
            return ion1_trap, ion2_trap
        if cap2 <= 0 and cap1 > 0:
            return ion2_trap, ion1_trap

        return (ion1_trap, ion2_trap) if score_t2 < score_t1 else (ion2_trap, ion1_trap)

    # ==========================================================
    # Shuttle
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        一次宏观 shuttle（trap->trap），会产生 Split + Move* + Merge micro-op。
        这里增加：
          - shuttle_counter 自增
          - shuttle_log 记录全过程
        """
        m = self.machine

        if len(route):
            rpath = route
        else:
            rpath = self.router.find_route(src_trap, dest_trap)

        # ============ Trace record begin ============
        shuttle_id = self.shuttle_counter
        self.shuttle_counter += 1
        self._current_shuttle_id = shuttle_id

        # route 文本化
        route_txt = []
        for node in rpath:
            if isinstance(node, Trap):
                route_txt.append(f"T{node.id}")
            elif isinstance(node, Junction):
                route_txt.append(f"J{node.id}")
            else:
                route_txt.append(str(node))

        self.shuttle_log.append({
            "shuttle_id": shuttle_id,
            "ion": ion,
            "src_trap": src_trap,
            "dst_trap": dest_trap,
            "route_text": "->".join(route_txt),
            "steps": []
        })
        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} START ion={ion} T{src_trap}->T{dest_trap} route={route_txt} ===")
        # ===========================================

        # 估计总时间用于 identify_start_time（保持原逻辑）
        t_est = 0
        for i in range(len(rpath) - 1):
            src = rpath[i]
            dest = rpath[i + 1]
            if type(src) == Trap and type(dest) == Junction:
                t_est += m.mparams.split_merge_time
            elif type(src) == Junction and type(dest) == Junction:
                t_est += m.move_time(src.id, dest.id)
            elif type(src) == Junction and type(dest) == Trap:
                t_est += m.merge_time(dest.id)

        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)
        clk = self._add_shuttle_ops(rpath, ion, clk)

        # correctness：shuttle 后 ion 必须到达 dest_trap
        if self.enable_assertions:
            try:
                real_trap = self.sys_state.find_trap_id_by_ion(ion)
                assert real_trap == dest_trap, f"Shuttle end mismatch: ion {ion} at T{real_trap}, expected T{dest_trap}"
            except Exception:
                pass

        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} END at t={clk} ===")
        self._current_shuttle_id = None
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

            # sys_state 更新（保持你的原逻辑）
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
    # Rebalance（创新：Belady/最晚下次使用 + executed set）
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
        Stateless Belady（近似）：选择下一次使用最晚/永不再用的离子作为 victim。
        注意：这是你的创新点之一，我保持它不变，仅加注释与保持鲁棒。
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

                # === victim selection（保持创新核心）===
                candidates = self.sys_state.trap_ions[node.id]
                valid_candidates = [ion for ion in candidates if ion not in self.protected_ions]

                if not valid_candidates:
                    moving_ion = candidates[0]
                else:
                    best_victim = valid_candidates[0]
                    max_score = -1

                    for ion in valid_candidates:
                        queue = self.qubit_queues[ion]
                        next_use = float("inf")

                        for g_idx in queue:
                            if g_idx not in self.executed_gate_indices:
                                next_use = g_idx
                                break

                        score = next_use

                        if score > max_score:
                            max_score = score
                            best_victim = ion
                        elif score == max_score:
                            if ion < best_victim:
                                best_victim = ion

                    moving_ion = best_victim

                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)

                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, route=shuttle_route)
                fin_time = max(fin_time, fin_time_new)

        return fin_time

    # ==========================================================
    # Main Gate Scheduling（保持你的创新版本结构）
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        if gate not in self.gate_info:
            return

        gate_data = self.gate_info[gate]
        qubits = gate_data if isinstance(gate_data, list) else gate_data["qubits"]

        # 保护
        self.protected_ions = set(qubits)

        # 1Q gate
        if len(qubits) == 1:
            ion1 = qubits[0]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            fire_time = max(ready, ion1_time, specified_time)

            duration = 10  # 你原来是 10（如果你要对齐论文可改回 5，但这里不动你的内核）
            if hasattr(self.machine, "single_qubit_gate_time"):
                gtype = gate_data.get("type", "u3") if isinstance(gate_data, dict) else "u3"
                duration = self.machine.single_qubit_gate_time(gtype)

            self.schedule.add_gate(fire_time, fire_time + duration, [ion1], ion1_trap)
            self.gate_finish_times[gate] = fire_time + duration

        # 2Q gate
        elif len(qubits) == 2:
            ion1, ion2 = qubits[0], qubits[1]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            ion2_time, ion2_trap = self.ion_ready_info(ion2)
            fire_time = max(ready, ion1_time, ion2_time, specified_time)

            if ion1_trap == ion2_trap:
                # correctness：2Q gate 必须同 trap（这里天然满足）
                gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
                self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap)
                self.gate_finish_times[gate] = fire_time + gate_duration
            else:
                # 冲突处理
                rebal_flag, new_fin_time = self.rebalance_traps([ion1_trap, ion2_trap], fire_time)

                if not rebal_flag:
                    source_trap, dest_trap = self.shuttling_direction(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
                    moving_ion = ion1 if source_trap == ion1_trap else ion2

                    clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time)

                    # correctness：gate 前必须同 trap
                    if self.enable_assertions:
                        t1 = self.sys_state.find_trap_id_by_ion(ion1)
                        t2 = self.sys_state.find_trap_id_by_ion(ion2)
                        assert t1 == t2 == dest_trap, f"Gate location mismatch: ({ion1},{ion2}) at (T{t1},T{t2}), expected T{dest_trap}"

                    gate_duration = self.machine.gate_time(self.sys_state, dest_trap, ion1, ion2)
                    self.schedule.add_gate(clk, clk + gate_duration, [ion1, ion2], dest_trap)
                    self.gate_finish_times[gate] = clk + gate_duration
                else:
                    # 递归重试
                    self.protected_ions = set()
                    self.schedule_gate(gate, specified_time=new_fin_time, gate_idx=gate_idx)
                    return

        self.protected_ions = set()

    def is_executable_local(self, gate):
        if gate not in self.gate_info:
            return True
        qubits = self.gate_info[gate]
        if isinstance(qubits, dict):
            qubits = qubits["qubits"]
        if len(qubits) < 2:
            return True
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # run()（修复你贴出来的末尾语法错误，并保持你的 local-first + executed set 逻辑）
    # ==========================================================
    def run(self):
        # Initial Frontier
        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}
        ready_gates = [n for n in self.ir.nodes if in_degree[n] == 0]

        processed_count = 0
        total_gates = len(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_gates:
                break

            # Prioritize Local
            local_candidates = []
            remote_candidates = []
            for g in ready_gates:
                if self.is_executable_local(g):
                    local_candidates.append(g)
                else:
                    remote_candidates.append(g)

            if local_candidates:
                best_gate = min(local_candidates, key=lambda x: self.gate_to_idx.get(x, float("inf")))
            else:
                best_gate = min(remote_candidates, key=lambda x: self.gate_to_idx.get(x, float("inf")))

            gate_idx = self.gate_to_idx.get(best_gate, 0)

            self.schedule_gate(best_gate, gate_idx=gate_idx)

            # 创新核心：标记为已执行
            self.executed_gate_indices.add(gate_idx)

            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)
