import math
import numpy as np

from schedule import Schedule


class AnalyzerKnobs:
    """
    Knobs for things NOT strictly fixed by the MUSS-TI paper text/tables.
    Keep paper-given constants inside Analyzer (do not tune them here).
    """
    def __init__(
        self,
        # Background Bi model
        bg_model: str = "linear",        # "none" | "linear"
        alpha_bg: float = 0.01157,           # strength for linear Bi

        # Heating injection normalization (paper doesn't explicitly say per-ion or per-chain accounting)
        inject_norm: str = "chain",      # "none" | "chain"  (chain => divide by chain length)
        swap_norm: str = "chain",        # "none" | "chain"

        # MOVE heating: distance-based (needs segment length), or constant per move op
        move_heat_use_distance: bool = True,
        move_heat_const: float = 0.1,    # fallback if not using distance

        # Gate env time mode: global finish time vs gate duration
        gate_env_time_mode: str = "global",  # "global" | "duration"

        # Debug
        debug_events: bool = False,
        debug_summary: bool = True,
    ):
        self.bg_model = bg_model
        self.alpha_bg = float(alpha_bg)

        self.inject_norm = inject_norm
        self.swap_norm = swap_norm

        self.move_heat_use_distance = move_heat_use_distance
        self.move_heat_const = float(move_heat_const)

        self.gate_env_time_mode = gate_env_time_mode

        self.debug_events = debug_events
        self.debug_summary = debug_summary


class Analyzer:
    """
    Analyzer aligned to MUSS-TI paper definitions:

    Paper-fixed (do NOT tune here):
    - Eq.1-like environment term: exp(-t/T1) - k*n_bar
    - Gate fidelity table constants: 1Q=0.9999, fiber=0.99, 2Q=1-eps*N^2
    - Heating increments table: Split=1, Merge=1, Swap=0.3, Move=0.1 per 2um (=> 0.05 per um)

    Not fully specified / implementation knobs:
    - Bi background model form/strength
    - whether heating injection is per-chain or per-ion accounting
    - MOVE heat distance model details (segment length)
    - gate env time using global time or duration
    """

    # ========== Paper-fixed constants ==========
    T1_US = 600 * 1e6
    K_HEATING = 0.001
    EPSILON_2Q = 1.0 / 25600.0

    HEAT_SPLIT = 1.0
    HEAT_MERGE = 1.0
    HEAT_SWAP = 0.3
    HEAT_MOVE_PER_UM = 0.1 / 2.0   # 0.05 per um

    FID_1Q = 0.9999
    FID_FIBER = 0.99

    def __init__(self, scheduler_obj, machine_obj, init_mapping, knobs: AnalyzerKnobs = None):
        self.scheduler = scheduler_obj
        self.schedule = scheduler_obj.schedule
        self.machine = machine_obj

        # initial trap->ions
        self.init_map = getattr(scheduler_obj, "init_map", init_mapping)

        # knobs
        self.knobs = knobs if knobs is not None else AnalyzerKnobs()

        # allow alpha_bg from machine params if not explicitly set
        if self.knobs.alpha_bg == 0.0:
            if hasattr(self.machine, "mparams") and hasattr(self.machine.mparams, "alpha_bg"):
                self.knobs.alpha_bg = float(self.machine.mparams.alpha_bg)

        # per-trap background factor B_i
        self.trap_bg = {t.id: 1.0 for t in self.machine.traps}

        # allocate heating (n_bar) for ions (safe upper bound)
        capacity = self.machine.traps[0].capacity if self.machine.traps else 20
        num_traps = len(self.machine.traps)
        max_ions = num_traps * capacity * 4
        self.ion_heating = {i: 0.0 for i in range(max_ions + 256)}

        # stats
        self.final_fidelity = 1.0
        self.prog_fin_time = 0.0
        self.op_count = {}

        self.gate_chain_lengths = []
        self._gate_mult = 1.0
        self._gate_cnt = 0
        self._gate_avg_n = []

        self._dyn_mult = 1.0
        self._dyn_cnt = 0
        self._dyn_min = 1.0

    # -------------------------
    # helpers
    # -------------------------
    def _seg_length_um(self, seg_id: int) -> float:
        if hasattr(self.machine, "get_segment_length_um"):
            try:
                return float(self.machine.get_segment_length_um(seg_id))
            except Exception:
                pass
        # fallback: assume seg_id==index
        try:
            return float(self.machine.segments[seg_id].length)
        except Exception:
            return float(getattr(self.machine.mparams, "segment_length_um", 53.0))

    def _avg_nbar(self, ions) -> float:
        if not ions:
            return 0.0
        return sum(self.ion_heating[i] for i in ions) / float(len(ions))

    def _apply_bg(self, trap_id: int, delta_avg_nbar: float):
        if self.knobs.bg_model == "none":
            return
        if self.knobs.bg_model == "linear":
            a = self.knobs.alpha_bg
            if a <= 0:
                return
            factor = 1.0 - a * float(delta_avg_nbar)
            factor = max(factor, 0.0)
            self.trap_bg[trap_id] *= factor
            self.trap_bg[trap_id] = min(max(self.trap_bg[trap_id], 0.0), 1.0)
            return
        # unknown model -> treat as none
        return

    def _env_fidelity(self, t_us: float, avg_nbar: float) -> float:
        f = math.exp(-float(t_us) / self.T1_US) - self.K_HEATING * float(avg_nbar)
        return max(f, 1e-15)

    def _gate_fidelity(self, chain_ions, is_2q: bool, is_fiber: bool, gate_start_us: float, gate_end_us: float, trap_id: int):
        N = len(chain_ions)
        if N <= 0:
            return 1e-15

        # gate intrinsic (paper)
        if is_fiber:
            f_g = self.FID_FIBER
        elif is_2q:
            f_g = 1.0 - self.EPSILON_2Q * (N ** 2)
        else:
            f_g = self.FID_1Q
        f_g = max(f_g, 1e-15)

        # env term (paper eq form, but time choice is a knob)
        avg_n = self._avg_nbar(chain_ions)
        if self.knobs.gate_env_time_mode == "duration":
            t_env = float(gate_end_us - gate_start_us)
        else:
            t_env = float(gate_end_us)  # global time since start

        f_env = self._env_fidelity(t_env, avg_n)

        # per-trap background Bi
        B = self.trap_bg.get(trap_id, 1.0)
        return max(B * f_env * f_g, 1e-15), avg_n, f_env, f_g, B

    def _dyn_event_mult(self, dt_us: float, delta_avg_nbar: float):
        # treat dynamics ops as additional env-like multiplicative penalty (implementation choice)
        f_dyn = self._env_fidelity(dt_us, delta_avg_nbar)
        self._dyn_mult *= f_dyn
        self._dyn_cnt += 1
        self._dyn_min = min(self._dyn_min, f_dyn)
        return f_dyn

    # -------------------------
    # main replay
    # -------------------------
    def move_check(self):
        self.op_count = {Schedule.Gate: 0, Schedule.Split: 0, Schedule.Move: 0, Schedule.Merge: 0}

        # replay state: trap->ions
        replay_traps = {t.id: [] for t in self.machine.traps}
        for t_id, ions in self.init_map.items():
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]

        # program finish time
        self.prog_fin_time = 0.0
        for ev in self.schedule.events:
            self.prog_fin_time = max(self.prog_fin_time, ev[3])

        acc = 1.0

        for ev in self.schedule.events:
            etype = ev[1]
            st, ed = float(ev[2]), float(ev[3])
            dt = max(ed - st, 0.0)
            info = ev[4]

            if etype in self.op_count:
                self.op_count[etype] += 1

            # ---- Gate ----
            if etype == Schedule.Gate:
                trap = info.get("trap", None)
                ions = info.get("ions", [])
                chain = replay_traps.get(trap, [])

                fid, avg_n, f_env, f_g, B = self._gate_fidelity(
                    chain_ions=chain,
                    is_2q=(len(ions) == 2),
                    is_fiber=info.get("is_fiber", False),
                    gate_start_us=st,
                    gate_end_us=ed,
                    trap_id=trap,
                )

                acc *= fid
                self._gate_mult *= fid
                self._gate_cnt += 1
                self._gate_avg_n.append(avg_n)

                if len(ions) == 2:
                    self.gate_chain_lengths.append(len(chain))

                if self.knobs.debug_events:
                    print("[DBG GATE]", "trap", trap, "L", len(chain),
                          "avg_n", round(avg_n, 6),
                          "B", round(B, 6),
                          "f_env", round(f_env, 6),
                          "f_g", round(f_g, 6),
                          "fid", round(fid, 6))

            # ---- Split ----
            elif etype == Schedule.Split:
                trap = info["trap"]
                moving_ions = info.get("ions", [])
                swap_cnt = int(info.get("swap_cnt", 0))

                chain = replay_traps.get(trap, [])
                L = len(chain)

                # chain-normalized injection knobs
                if self.knobs.inject_norm == "chain" and L > 0:
                    d_split = self.HEAT_SPLIT / float(L)
                else:
                    d_split = self.HEAT_SPLIT

                if L > 0:
                    for ion in chain:
                        self.ion_heating[ion] += d_split
                    self._apply_bg(trap, d_split)

                # swap heating
                if swap_cnt > 0:
                    if self.knobs.swap_norm == "chain" and L > 0:
                        d_swap = (self.HEAT_SWAP * swap_cnt) / float(L)
                    else:
                        d_swap = (self.HEAT_SWAP * swap_cnt)
                    if L > 0:
                        for ion in chain:
                            self.ion_heating[ion] += d_swap
                else:
                    d_swap = 0.0

                # dynamics multiplicative penalty
                f_dyn = self._dyn_event_mult(dt, d_split + d_swap)
                acc *= f_dyn

                # remove moving ions from trap chain
                for ion in moving_ions:
                    if ion in chain:
                        chain.remove(ion)

                if self.knobs.debug_events:
                    print("[DBG SPLIT]", "trap", trap, "L", L,
                          "dt", dt,
                          "d_split", round(d_split, 6),
                          "d_swap", round(d_swap, 6),
                          "f_dyn", round(f_dyn, 6))

            # ---- Move ----
            elif etype == Schedule.Move:
                ions = info.get("ions", [])
                dst_seg = info.get("dest_seg", None)

                if self.knobs.move_heat_use_distance:
                    dist_um = self._seg_length_um(dst_seg) if dst_seg is not None else float(getattr(self.machine.mparams, "segment_length_um", 53.0))
                    heat = dist_um * self.HEAT_MOVE_PER_UM
                else:
                    heat = self.knobs.move_heat_const

                for ion in ions:
                    self.ion_heating[ion] += heat

                f_dyn = self._dyn_event_mult(dt, heat)
                acc *= f_dyn

                if self.knobs.debug_events:
                    print("[DBG MOVE]", "dst_seg", dst_seg,
                          "dt", dt, "heat", round(heat, 6),
                          "f_dyn", round(f_dyn, 6))

            # ---- Merge ----
            elif etype == Schedule.Merge:
                trap = info["trap"]
                incoming = info.get("ions", [])

                new_chain = replay_traps.get(trap, []) + incoming
                replay_traps[trap] = new_chain
                L = len(new_chain)

                if self.knobs.inject_norm == "chain" and L > 0:
                    d_merge = self.HEAT_MERGE / float(L)
                else:
                    d_merge = self.HEAT_MERGE

                if L > 0:
                    for ion in new_chain:
                        self.ion_heating[ion] += d_merge
                    self._apply_bg(trap, d_merge)

                f_dyn = self._dyn_event_mult(dt, d_merge)
                acc *= f_dyn

                # keep merge equalization (your confirmed behavior)
                if new_chain:
                    avg_h = self._avg_nbar(new_chain)
                    for ion in new_chain:
                        self.ion_heating[ion] = avg_h

                if self.knobs.debug_events:
                    print("[DBG MERGE]", "trap", trap, "L", L,
                          "dt", dt, "d_merge", round(d_merge, 6),
                          "f_dyn", round(f_dyn, 6))

        self.final_fidelity = acc
        self._print_stats()

    def _print_stats(self):
        print(f"Program Finish Time: {self.prog_fin_time} us")
        print("OPCOUNTS",
              "Gate:", self.op_count.get(Schedule.Gate, 0),
              "Split:", self.op_count.get(Schedule.Split, 0),
              "Move:", self.op_count.get(Schedule.Move, 0),
              "Merge:", self.op_count.get(Schedule.Merge, 0))

        if self.gate_chain_lengths:
            lens = np.array(self.gate_chain_lengths, dtype=float)
            print("\nTwo-qubit gate chain statistics")
            print(f"Mean: {np.mean(lens)} Max: {np.max(lens)}")

        print(f"Fidelity: {self.final_fidelity}")

        if self.knobs.debug_summary:
            if self._gate_cnt > 0:
                avg_gate_n = float(np.mean(self._gate_avg_n)) if self._gate_avg_n else 0.0
                print(f"[DBG SUMMARY] gates={self._gate_cnt}  gate_mult={self._gate_mult:.6g}  avg_gate_nbar={avg_gate_n:.4f}")
            if self._dyn_cnt > 0:
                print(f"[DBG SUMMARY] dyn_ops={self._dyn_cnt}  dyn_mult={self._dyn_mult:.6g}  min_dyn={self._dyn_min:.6g}")
            worst = min(self.trap_bg.items(), key=lambda x: x[1])
            print(f"[DBG SUMMARY] worst_B: Trap {worst[0]} -> {worst[1]:.6f} (alpha_bg={self.knobs.alpha_bg})")

    def analyze_and_return(self):
        self.move_check()
        if hasattr(self.scheduler, "shuttle_counter"):
            shuttle_count = int(getattr(self.scheduler, "shuttle_counter"))
        else:
            shuttle_count = int(self.op_count.get(Schedule.Split, 0))
        return {"fidelity": self.final_fidelity, "total_shuttle": shuttle_count, "time": self.prog_fin_time}
