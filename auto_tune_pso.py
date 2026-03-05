import sys
import os
import numpy as np
import traceback
from pyswarm import pso

# 导入项目模块
from parse import InputParse
from machine import MachineParams
from test_machines import test_trap_2x3
from mappers import QubitMapSABRE6
from muss_schedule4 import MUSSSchedule
from analyzer import Analyzer

# === 配置区域 ===
QASM_FILE = "./programs/QFT32.qasm"
MACHINE_TYPE = "G2x3"
NUM_IONS = 8
GATE_TYPE = "FM"
SWAP_TYPE = "GateSwap"

# 权重与参考值 (参考 v2)
REF_FIDELITY = 1.5e-12
REF_SHUTTLE = 200
REF_TIME = 36000
W_FIDELITY = 50.0
W_SHUTTLE = 1.0
W_TIME = 0.5

def evaluate_schedule(params):
    lambda_val, beta_val = params
    try:
        # 1. 准备环境
        mpar = MachineParams()
        mpar.split_merge_time, mpar.shuttle_time = 80, 5
        mpar.ion_swap_time = 40
        mpar.junction2_cross_time = 5
        mpar.junction3_cross_time = 5
        mpar.junction4_cross_time = 5
        mpar.gate_type, mpar.swap_type = GATE_TYPE, SWAP_TYPE
        mpar.T1, mpar.k_heating, mpar.epsilon = 600e6, 0.001, 1.0/25600.0

        machine = test_trap_2x3(NUM_IONS, mpar)
        ip = InputParse()
        ip.parse_ir(QASM_FILE)
        
        # 运行 Mapper
        mapper = QubitMapSABRE6(ip, machine)
        raw_mapping = mapper.compute_mapping()
        
        # === 核心修复：转换映射格式 ===
        # MUSSSchedule 期望格式: {trap_id: [ion_ids]}
        init_layout = {t.id: [] for t in machine.traps}
        for q, t in raw_mapping.items():
            if t in init_layout:
                init_layout[t].append(q)
        
        # 2. 实例化调度器
        scheduler = MUSSSchedule(ip.gate_graph, ip.cx_gate_map, machine, init_layout, 1, 0, 0)
        scheduler.nbar_penalty_lambda = lambda_val
        scheduler.rebalance_heat_beta = beta_val
        
        # 3. 运行调度
        # 注意：v4 Fixed 内部自带热量追踪，不需要外部 analyzer 注入
        scheduler.run()
        
        # 4. 分析结果
        analyzer = Analyzer(scheduler, machine, init_layout)
        result = analyzer.analyze_and_return() # 确保 analyzer.py 有此方法
        
        fidelity = result['fidelity']
        shuttle = result['total_shuttle']
        time = result['time']
        
        # 5. 计算 Cost
        if fidelity <= 1e-25: fidelity = 1e-25
        cost_fid = (np.log10(REF_FIDELITY) - np.log10(fidelity)) * W_FIDELITY
        cost_shuttle = (shuttle / REF_SHUTTLE) * W_SHUTTLE
        cost_time = (time / REF_TIME) * W_TIME
        total_cost = cost_fid + cost_shuttle + cost_time
        
        print(f"Eval Success: λ={lambda_val:.4f}, β={beta_val:.2f} -> Fid={fidelity:.2e}, Cost={total_cost:.4f}")
        return total_cost

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 1e9

if __name__ == "__main__":
    lb = [0.001, 0.5]
    ub = [0.05,  10]

    print(f"Starting Single Test...")
    test_res = evaluate_schedule([0.005, 1.0])
    
    if test_res < 1e8:
        print("\nTest passed! Starting PSO Optimization...")
        # 为了速度，你可以先用较小的 swarm 和 iter
        best_params, best_cost = pso(evaluate_schedule, lb, ub, swarmsize=8, maxiter=10)
        print("-" * 30)
        print(f"FINAL BEST PARAMS: lambda={best_params[0]:.6f}, beta={best_params[1]:.4f}")
        print(f"FINAL BEST COST: {best_cost:.4f}")
    else:
        print("\nTest still failing. Check the Traceback above.")
