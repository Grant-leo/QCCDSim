"""
Machine class definition

本文件定义了离子阱（Trap）- 传输段（Segment）- 交叉点（Junction）的机器模型，并提供：
1) 构建机器拓扑（graph）接口：add_trap / add_segment / add_junction
2) 调度器需要的时间模型：
   - gate_time：两比特门时间（按 gate_type）
   - split_time / merge_time：分裂/合并时间与 split-swap 统计
   - move_time：按物理距离与速度计算的移动时间（Table 1: 2 μm/us）
   - junction_cross_time：穿越 junction 的额外时间
3) 论文复现所需的“物理量参数化”（论文没给的外提为可调参数）：
   - segment_length_um：segment 物理长度（um）
   - move_speed_um_per_us：移动速度（um/us）
   - inter_ion_spacing_um：阱内离子间距（um）
   - alpha_bg：背景 Bi 模型强度（供 Analyzer 用）

注意：
- gate_time 对 FM 直接返回固定 40(us)，这是为了对齐 MUSS-TI 论文评估；保留该行为。
- 本文件尽量兼容旧项目：如果没传 mparams，会自动创建默认 MachineParams。
"""

import networkx as nx


# =========================
#   基本结构：Trap / Segment / Junction
# =========================
class Trap:
    """离子阱：容纳一条离子链（chain）。orientation 记录每条 segment 在阱的左右侧。"""

    def __init__(self, idx, capacity: int):
        self.id = idx
        self.capacity = capacity
        self.ions = []  # （旧逻辑保留）可能用于一些调试/展示
        # orientation: seg_id -> "L"/"R"
        # 表示该 segment 连接到 trap 的左端或右端，用于 split 时判断链端离子
        self.orientation = {}

    def show(self):
        return "T" + str(self.id)


class Segment:
    """传输段：连接两个节点（Trap 或 Junction）的边对象，带物理长度（um）。"""

    def __init__(self, idx, capacity: int, length_um: float):
        self.id = idx
        self.capacity = capacity
        # 物理长度（um）
        self.length = float(length_um)
        self.ions = []  # （旧逻辑保留）


class Junction:
    """交叉点：用于多段 segment 的连接（图的节点）。"""

    def __init__(self, idx):
        self.id = idx
        self.objs = []  # （旧逻辑保留）

    def show(self):
        return "J" + str(self.id)


# =========================
#   参数容器：MachineParams
# =========================
class MachineParams:
    """
    参数容器（建议固定“论文明确给出”的默认值；论文没给的做显式 knob）

    论文明确给出的（MUSS-TI Table 1）：
      - split_merge_time (us) = 80
      - ion_swap_time (us)    = 40
      - move_speed_um_per_us  = 2.0 (um/us)
      - junction{2,3,4}_cross_time (us) = 5
      - shuttle_time (us) = 5 （旧字段，兼容回退）

    论文没明确钉死 / 实现需要外提的 knob：
      - segment_length_um (um)        ：每条 segment 默认长度（影响 move_time 与 move heating）
      - inter_ion_spacing_um (um)     ：阱内离子间距（影响 Duan/Trout/PM 的 gate_time 距离项）
      - alpha_bg                      ：背景 Bi 模型强度（Analyzer 用）
      - gate_type / swap_type         ：门/交换实现类型
    """

    def __init__(self):
        # ===== MUSS-TI Table 1 (paper-fixed defaults) =====
        self.split_merge_time = 80
        self.shuttle_time = 5
        self.ion_swap_time = 40

        self.junction2_cross_time = 5
        self.junction3_cross_time = 5
        self.junction4_cross_time = 5

        self.move_speed_um_per_us = 2.0  # 2 μm/us

        # ===== Not explicitly fixed by paper / implementation knobs =====
        self.segment_length_um = 80.0          # 每条 segment 默认长度（um）
        self.inter_ion_spacing_um = 1.0        # 阱内离子间距（um）
        self.alpha_bg = 0.0                    # 背景 Bi 强度（Analyzer 用）

        # ===== gate/swap type flags =====
        self.gate_type = "PM"                  # "FM"/"PM"/"Duan"/"Trout"
        self.swap_type = "GateSwap"            # "GateSwap"/"IonSwap"


# =========================
#   Machine：机器模型主体
# =========================
class Machine:
    """
    Machine 用 networkx.Graph() 存拓扑：
    - 节点：Trap 或 Junction
    - 边：Segment（作为 edge 属性 "seg" 挂在图上）

    额外维护：
    - traps / segments / junctions 列表（方便遍历）
    - segments_by_id：避免 seg_id != index 时访问错误
    - dist_cache：trap-to-trap 最短路径 hop 数缓存
    """

    def __init__(self, mparams=None):
        # 允许 mparams=None（兼容旧代码）
        self.mparams = mparams if mparams is not None else MachineParams()

        self.graph = nx.Graph()
        self.traps = []
        self.segments = []
        self.junctions = []

        # seg_id -> Segment object（避免 seg_id != index 的错误）
        self.segments_by_id = {}

        # trap-to-trap distance cache (hop count)
        self.dist_cache = {}

    # -------------------------
    # Build graph API
    # -------------------------
    def add_trap(self, idx, capacity: int):
        """添加一个 Trap 节点。"""
        new_trap = Trap(idx, capacity)
        self.traps.append(new_trap)
        self.graph.add_node(new_trap)
        return new_trap

    def add_junction(self, idx):
        """添加一个 Junction 节点。"""
        new_junct = Junction(idx)
        self.junctions.append(new_junct)
        self.graph.add_node(new_junct)
        return new_junct

    def add_segment(self, idx, obj1, obj2, orientation="L"):
        """
        添加一条 Segment 边，并把 Segment 对象挂到 graph edge 的属性 "seg" 上。

        参数：
        - idx：segment id
        - obj1 / obj2：图的两个端点（Trap 或 Junction）
        - orientation：仅对 Trap 有意义。记录该 segment 在 trap 的左/右侧。

        说明：
        - segment 长度来自 mparams.segment_length_um（可调 knob）
        - Segment capacity 这里保持为 16（沿用旧实现；如需也可外提为 knob）
        """
        seg_len = float(getattr(self.mparams, "segment_length_um", 10.0))
        new_seg = Segment(idx, 16, seg_len)

        self.segments.append(new_seg)
        self.segments_by_id[new_seg.id] = new_seg

        # Trap 需要记录左右侧
        if isinstance(obj1, Trap):
            obj1.orientation[new_seg.id] = orientation

        # 旧实现里不允许 Junction->Trap 作为 obj1（保持一致）
        if isinstance(obj1, Junction) and isinstance(obj2, Trap):
            raise AssertionError("add_segment junction->trap is not allowed by this API.")

        self.graph.add_edge(obj1, obj2, seg=new_seg)

    def get_segment_length_um(self, seg_id: int) -> float:
        """
        给 Analyzer/调度器调用：返回 segment 的物理长度（um）
        优先用 segments_by_id；否则回退假设 seg_id==index；再回退默认值。
        """
        if seg_id in self.segments_by_id:
            return float(self.segments_by_id[seg_id].length)
        try:
            return float(self.segments[seg_id].length)
        except Exception:
            return float(getattr(self.mparams, "segment_length_um", 10.0))

    def add_comm_capacity(self, val: int):
        """给所有 trap 增加容量（旧接口保留）。"""
        for t in self.traps:
            t.capacity += val

    def print_machine_stats(self):
        """旧接口保留（原项目可能调用）。"""
        if self.traps:
            _ = self.traps[0].capacity

    # -------------------------
    # Gate / Trap 操作时间
    # -------------------------
    def gate_time(self, sys_state, trap_id: int, ion1: int, ion2: int) -> int:
        """
        计算两比特门时间（us）。

        对齐论文评估：
        - gate_type == "FM" 时直接返回固定 40us（MUSS-TI Table 1）
        其它 gate_type 保留原经验公式（与离子距离相关）。

        sys_state 依赖：
        - sys_state.trap_ions[trap_id]：该 trap 内离子链的顺序列表
        """
        assert ion1 != ion2
        mp = self.mparams

        p1 = sys_state.trap_ions[trap_id].index(ion1)
        p2 = sys_state.trap_ions[trap_id].index(ion2)

        # 离子间距（um），原来写死 1，这里可配置
        d_const = float(getattr(mp, "inter_ion_spacing_um", 1.0))
        ion_dist = abs(p1 - p2) * d_const

        gate_type = getattr(mp, "gate_type", "FM")

        if gate_type == "Duan":
            t = -22 + 100 * ion_dist
        elif gate_type == "Trout":
            t = 10 + 38 * ion_dist
        elif gate_type == "FM":
            return 40
        elif gate_type == "PM":
            t = 160 + 5 * ion_dist
        else:
            raise AssertionError(f"Unsupported gate_type: {gate_type}")

        t = max(t, 1)
        return int(t)

    def split_time(self, sys_state, trap_id: int, seg_id: int, ion1: int):
        """
        计算 split_time，并给出 split swap 的统计信息（供 schedule/event 记录）
        保留原逻辑不变。

        返回：
          (split_estimate,
           split_swap_count,
           split_swap_hops,
           i1, i2,
           ion_swap_hops)

        说明：
        - split 的离子必须在链端，否则需要 swap 到链端（swap_type 决定实现方式）
        - GateSwap：用门实现 swap（3 * gate_time）
        - IonSwap：用 split/move/merge 组合实现 swap（更“物理”）
        """
        t = self.traps[trap_id]
        split_estimate = 0
        split_swap_count = 0
        ion_swap_hops = 0
        split_swap_hops = 0
        i1 = 0
        i2 = 0

        # 要 split 的离子必须在链端，否则需要 swap 到链端
        if t.orientation[seg_id] == "L":
            ion2 = sys_state.trap_ions[trap_id][0]   # 左端
        else:
            ion2 = sys_state.trap_ions[trap_id][-1]  # 右端

        if ion1 == ion2:
            split_estimate = self.mparams.split_merge_time
            split_swap_count = 0
            split_swap_hops = 0
        else:
            mp = self.mparams
            swap_type = getattr(mp, "swap_type", "GateSwap")

            if swap_type == "GateSwap":
                # 用门实现 swap：3 * gate_time（旧逻辑保留）
                swap_est = 3 * self.gate_time(sys_state, trap_id, ion1, ion2)
                split_estimate = swap_est + self.mparams.split_merge_time
                p1 = sys_state.trap_ions[trap_id].index(ion1)
                p2 = sys_state.trap_ions[trap_id].index(ion2)
                split_swap_count = 1
                split_swap_hops = abs(p1 - p2)
                i1 = ion1
                i2 = ion2

            elif swap_type == "IonSwap":
                # 通过 ion moves + splits/merges 实现 swap（旧逻辑保留）
                p1 = sys_state.trap_ions[trap_id].index(ion1)
                p2 = sys_state.trap_ions[trap_id].index(ion2)
                num_hops = abs(p1 - p2)

                swap_est = num_hops * self.mparams.split_merge_time         # n splits
                swap_est += (num_hops - 1) * self.mparams.split_merge_time  # n-1 merges
                swap_est += self.mparams.ion_swap_time * num_hops           # n moves
                split_estimate = swap_est

                ion_swap_hops = num_hops
            else:
                raise AssertionError(f"Unsupported swap_type: {swap_type}")

        return int(split_estimate), split_swap_count, split_swap_hops, i1, i2, ion_swap_hops

    def merge_time(self, trap_id: int) -> int:
        """合并时间（us），论文 Table 1：与 split 同为 split_merge_time。"""
        return int(self.mparams.split_merge_time)

    # -------------------------
    # Move / Junction 时间
    # -------------------------
    def move_time(self, seg1_id: int, seg2_id: int) -> int:
        """
        按论文 Table 1 的速度模型计算 Move 时间：
            speed = move_speed_um_per_us（默认 2.0）
            time_us = distance_um / speed

        注意：
        - schedule 的 Move event 是 segment-to-segment 的 hop。
        - 这里用“目标 segment 的 length”作为本次 hop 的物理距离（简单且可校准）。
        - 若 move_speed_um_per_us 未设置，回退到旧字段 shuttle_time（兼容旧实现）。
        """
        speed = getattr(self.mparams, "move_speed_um_per_us", None)
        if speed is None:
            return int(getattr(self.mparams, "shuttle_time", 5))

        dist_um = self.get_segment_length_um(seg2_id)
        try:
            t_us = float(dist_um) / float(speed)
        except Exception:
            t_us = float(getattr(self.mparams, "shuttle_time", 5))

        return int(round(max(t_us, 1.0)))

    def junction_cross_time(self, junct) -> int:
        """
        junction crossing 的额外时间，按 junction 度数选择参数：
        - degree=2 用 junction2_cross_time
        - degree=3 用 junction3_cross_time
        - degree=4 用 junction4_cross_time
        """
        deg = self.graph.degree(junct)
        if deg == 2:
            return int(self.mparams.junction2_cross_time)
        elif deg == 3:
            return int(self.mparams.junction3_cross_time)
        elif deg == 4:
            return int(self.mparams.junction4_cross_time)
        else:
            raise AssertionError(f"Unsupported junction degree: {deg}")

    # -------------------------
    # 其它接口（保留原有功能）
    # -------------------------
    def single_qubit_gate_time(self, gate_type) -> int:
        """1Q gate 时间（旧实现保留，固定很小的常数）。"""
        return 5

    def precompute_distances(self):
        """
        预计算 trap-to-trap 的最短路径长度（按 graph hop 数）。
        hop 是 networkx shortest_path_length 的结果（节点之间的边数）。

        注意：graph 的节点是 Trap/Junction 对象，因此需要通过对象来索引 all_pairs_shortest_path_length。
        """
        self.dist_cache = {}
        id_map = {t.id: t for t in self.traps}

        try:
            all_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
        except Exception as e:
            print("Warning: Failed to compute distances:", e)
            all_paths = {}

        for id1 in id_map:
            for id2 in id_map:
                t1 = id_map[id1]
                t2 = id_map[id2]

                if t1 == t2:
                    self.dist_cache[(id1, id2)] = 0
                elif t1 in all_paths and t2 in all_paths.get(t1, {}):
                    self.dist_cache[(id1, id2)] = all_paths[t1][t2]
                else:
                    # 不连通或异常情况：用一个很大的值
                    self.dist_cache[(id1, id2)] = 1000
