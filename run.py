import sys
from parse import InputParse
from mappers import *
from machine import Machine, MachineParams, Trap, Segment
from ejf_schedule import Schedule, EJFSchedule
from analyzer import *
from test_machines import *
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

# 导入两个版本的 MUSSSchedule，并取别名避免冲突
from muss_schedule2 import MUSSSchedule as MUSSScheduleV2  # 论文版本调度
from muss_schedule3 import MUSSSchedule as MUSSScheduleV3  # 创新版本调度
from muss_schedule4 import MUSSSchedule as MUSSScheduleV4  # 创新版本调度基础上的降低加热版本。

np.random.seed(12345)

# Command line args
openqasm_file_name = sys.argv[1]
machine_type       = sys.argv[2]
num_ions_per_region = int(sys.argv[3])
mapper_choice      = sys.argv[4]
reorder_choice     = sys.argv[5]

serial_trap_ops = int(sys.argv[6])   # 阱内操作是否串行
serial_comm     = int(sys.argv[7])   # 通信是否串行
serial_all      = int(sys.argv[8])   # 所有操作是否全串行

gate_type = sys.argv[9]     # PM / FM 等
swap_type = sys.argv[10]    # GateSwap / IonSwap 等

# 新增：第11、12个参数用于选择调度器家族和版本（可缺省）
sched_family  = sys.argv[11].upper() if len(sys.argv) >= 12 else "MUSS"
sched_version = sys.argv[12].upper() if len(sys.argv) >= 13 else "V3"   # 默认用 v3（创新版）

# ────────────────────────────────────────────────
#              Machine 参数（MUSS-TI Table 1）
# ────────────────────────────────────────────────
mpar = MachineParams()
mpar.split_merge_time       = 80
mpar.shuttle_time           = 5
mpar.ion_swap_time          = 40
mpar.junction2_cross_time   = 5
mpar.junction3_cross_time   = 5
mpar.junction4_cross_time   = 5

# 保真度相关（analyzer 会使用）
mpar.T1         = 600e6         # us
mpar.k_heating  = 0.001
mpar.epsilon    = 1.0 / 25600.0

mpar.gate_type  = gate_type
mpar.swap_type  = swap_type

machine_model = "MUSS_Params"

# ────────────────────────────────────────────────
#                    打印基本信息
# ────────────────────────────────────────────────
print("Simulation")
print("Program:     ", openqasm_file_name)
print("Machine:     ", machine_type)
print("Model:       ", machine_model)
print("Ions:        ", num_ions_per_region)
print("Mapper:      ", mapper_choice)
print("Reorder:     ", reorder_choice)
print("SerialTrap:  ", serial_trap_ops)
print("SerialComm:  ", serial_comm)
print("SerialAll:   ", serial_all)
print("Gatetype:    ", gate_type)
print("Swaptype:    ", swap_type)
print("Scheduler Family:", sched_family)
print("Scheduler Version:", sched_version)

# ────────────────────────────────────────────────
#                    创建测试机器
# ────────────────────────────────────────────────
if machine_type == "G2x3":
    m = test_trap_2x3(num_ions_per_region, mpar)
elif machine_type == "G2x2":
    m = test_trap_2x2(num_ions_per_region, mpar)
elif machine_type == "L6":
    m = make_linear_machine(6, num_ions_per_region, mpar)
elif machine_type == "H6":
    m = make_single_hexagon_machine(num_ions_per_region, mpar)
else:
    print(f"不支持的机器类型: {machine_type}")
    sys.exit(1)

# ────────────────────────────────────────────────
#                    解析电路
# ────────────────────────────────────────────────
ip = InputParse()
ip.parse_ir(openqasm_file_name)
ip.visualize_graph("visualize_graph_2.gexf")

qc = QuantumCircuit.from_qasm_file(openqasm_file_name)
dag = circuit_to_dag(qc)
# dag_drawer(dag, filename=f"{openqasm_file_name[:-5]}.svg")   # 可选

print("parse object map:")
print(ip.cx_gate_map)
print("parse object graph:")
print(ip.gate_graph)

# ────────────────────────────────────────────────
#                    初始映射
# ────────────────────────────────────────────────
if mapper_choice == "LPFS":
    qm = QubitMapLPFS(ip, m)
elif mapper_choice == "Agg":
    qm = QubitMapAgg(ip, m)
elif mapper_choice == "Random":
    qm = QubitMapRandom(ip, m)
elif mapper_choice == "PO":
    qm = QubitMapPO(ip, m)
elif mapper_choice == "Greedy":
    qm = QubitMapGreedy(ip, m)
elif mapper_choice == "Trivial":
    qm = QubitMapTrivial(ip, m)
elif mapper_choice == "SABRE":
    # 根据调度器版本选择对应的 SABRE 映射变体
    if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
        if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
            print("→ Using SABRE3 mapper (matches muss_schedule2 paper version)")
            qm = QubitMapSABRE3(ip, m)
        elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
            print("→ Using SABRE6 mapper (matches muss_schedule3 improved version)")
            qm = QubitMapSABRE6(ip, m)
        elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
            print("→ Using SABRE6 mapper (matches muss_schedule4 improved version)")
            qm = QubitMapSABRE6(ip, m)
        else:
            print(f"Warning: Unknown scheduler version '{sched_version}', fallback to SABRE3")
            qm = QubitMapSABRE3(ip, m)
    else:
        # 非 MUSS 家族调度器时，使用默认 SABRE3
        print("→ Using default SABRE3 mapper (non-MUSS scheduler)")
        qm = QubitMapSABRE3(ip, m)
else:
    print(f"Error: Unsupported mapper choice '{mapper_choice}'")
    sys.exit(1)

mapping = qm.compute_mapping()

# ────────────────────────────────────────────────
#                    阱内重排序
# ────────────────────────────────────────────────
if mapper_choice == "Greedy":
    init_qubit_layout = mapping
else:
    qo = QubitOrdering(ip, m, mapping)
    if reorder_choice == "Naive":
        init_qubit_layout = qo.reorder_naive()
    elif reorder_choice == "Fidelity":
        init_qubit_layout = qo.reorder_fidelity()
    else:
        print(f"不支持的 reorder: {reorder_choice}")
        sys.exit(1)

print("init_qubit_layout:")
print(init_qubit_layout)

# ────────────────────────────────────────────────
#                    调度阶段
# ────────────────────────────────────────────────
print(f"Using {sched_family} Scheduler ({sched_version}) with {mapper_choice} Mapping")

if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
    if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
        print(" muss_schedule2.py old_vision")
        scheduler = MUSSScheduleV2(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
        print(" muss_schedule3.py new_vision")
        scheduler = MUSSScheduleV3(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
        print(" muss_schedule4.py new_vision")
        scheduler = MUSSScheduleV4(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    else:
        print(f"error：不支持的调度器版本 {sched_version}，支持：V2 / V3 / V4")
        sys.exit(1)
else:
    print("→ 回退使用 EJF 调度器")
    scheduler = EJFSchedule(
        ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
        serial_trap_ops, serial_comm, serial_all
    )

scheduler.run()

# ────────────────────────────────────────────────
#                    分析与输出
# ────────────────────────────────────────────────
analyzer = Analyzer(scheduler, m, init_qubit_layout)
analyzer.move_check()

if hasattr(scheduler, "split_swap_counter"):
    print("SplitSWAP:", scheduler.split_swap_counter)

print("----------------")
