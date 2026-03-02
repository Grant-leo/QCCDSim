import math
from utils import *          
from machine_state import *   
from schedule import *        

class Analyzer:
    """
    分析器类，用于重放调度事件，计算保真度、加热量等性能指标。
    基于 MUSS-TI 论文的参数和模型。
    """
    def __init__(self, scheduler_obj, machine_obj, init_mapping):
        """
        初始化分析器。
        """
        self.scheduler = scheduler_obj                     
        self.schedule = scheduler_obj.schedule             
        self.machine = machine_obj                         

        # 合法化初始映射
        if hasattr(scheduler_obj, "init_map"):             
            self.init_map = scheduler_obj.init_map         
        else:
            self.init_map = init_mapping                   

        # === MUSS-TI Paper Parameters (Table 1 & Eq 1) ===
        self.T1 = 600 * 1e6                                # 相干时间 600 秒
        self.k = 0.001                                      # 加热率系数 k
        self.epsilon = 1 / 25600.0                          # 保真度衰减系数 ε

        # Heating Quanta per Operation (Table 1)
        self.HEAT_SPLIT = 1.0                               # 拆分加热
        self.HEAT_MERGE = 1.0                               # 合并加热
        self.HEAT_MOVE = 0.1                                # 移动加热
        self.HEAT_SWAP = 0.3                                # 交换加热

        # Base Fidelities
        self.FID_FIBER = 0.99                               # 光纤门基础保真度

        # === 离子级热量追踪 ===
        capacity = self.machine.traps[0].capacity if self.machine.traps else 20   
        num_traps = len(self.machine.traps)                                       
        max_ions = num_traps * capacity * 2                                       

        # key: ion_id, value: accumulated n_bar
        self.ion_heating = {i: 0.0 for i in range(max_ions + 50)}                 
        self.gate_chain_lengths = []                                               

        # PSO 接口所需的属性初始化
        self.final_fidelity = 0.0
        self.prog_fin_time = 0.0
        self.op_count = {}

    def compute_gate_fidelity(self, chain_ions, is_fiber=False):
        """
        根据 MUSS-TI 论文公式计算单门保真度。
        """
        N = len(chain_ions)                               
        if N == 0:
            return 0.0

        # 1. Chain Length Penalty (epsilon * N^2)
        if is_fiber:
            f_gate = self.FID_FIBER                        
        else:
            f_gate = 1.0 - self.epsilon * (N**2)           

        # 2. Heating Penalty (k * n_bar)
        total_n = sum(self.ion_heating[ion] for ion in chain_ions)   
        avg_n_bar = total_n / N                                       

        heating_penalty = self.k * avg_n_bar                          
        b_i = math.exp(-heating_penalty)                              

        return f_gate * b_i                                            

    def move_check(self):
        """
        严格重放调度事件，计算最终性能指标。
        """
        # 初始化操作计数和总时间统计 (转为 self 属性)
        self.op_count = {Schedule.Gate: 0, Schedule.Split: 0, Schedule.Move: 0, Schedule.Merge: 0}
        op_times = {Schedule.Gate: 0.0, Schedule.Split: 0.0, Schedule.Move: 0.0, Schedule.Merge: 0.0}

        # === 1. 重建初始物理分布 ===
        replay_traps = {t.id: [] for t in self.machine.traps}          
        for t_id, ions in self.init_map.items():                        
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]                            

        self.prog_fin_time = 0.0                                             
        log_fidelity = 0.0                                              

        # 计算程序结束时间
        for event in self.schedule.events:
            self.prog_fin_time = max(self.prog_fin_time, event[3])                

        # === 2. 事件重演 ===
        for event in self.schedule.events:
            etype = event[1]                                             
            start_t, end_t = event[2], event[3]                          
            duration = end_t - start_t                                   
            info = event[4]                                              

            if etype in self.op_count:
                self.op_count[etype] += 1                                     
                op_times[etype] += duration                              

            # --- GATE ---
            if etype == Schedule.Gate:
                ions = info["ions"]                                      
                trap = info["trap"]                                      

                if len(ions) == 2:                                       
                    chain_ions = replay_traps.get(trap, [])              
                    is_fiber = info.get("is_fiber", False)               

                    fid = self.compute_gate_fidelity(chain_ions, is_fiber)   
                    self.gate_chain_lengths.append(len(chain_ions))           
                    
                    if len(chain_ions) > 0:
                        total_n = sum(self.ion_heating[ion] for ion in chain_ions)
                        avg_n = total_n / len(chain_ions)
                        if not hasattr(self, 'chain_nbar_records'):
                            self.chain_nbar_records = []                             
                        self.chain_nbar_records.append((trap, avg_n, len(chain_ions), fid))
                    
                    if fid > 0:
                        log_fidelity += math.log(fid)                                 
                    else:
                        log_fidelity += -999.0                                        

            # --- SPLIT ---
            elif etype == Schedule.Split:
                trap = info["trap"]                                        
                moving_ions = info["ions"]                                 

                # Split 增加热量
                for ion in replay_traps[trap]:
                    self.ion_heating[ion] += self.HEAT_SPLIT               

                # Swap Penalty
                total_swaps = info.get("ion_hops", 0) + info.get("swap_hops", 0)   
                if total_swaps > 0:
                    for ion in replay_traps[trap]:
                        self.ion_heating[ion] += self.HEAT_SWAP * total_swaps      

                # 更新离子位置
                for ion in moving_ions:
                    if ion in replay_traps[trap]:
                        replay_traps[trap].remove(ion)                     

            # --- MOVE ---
            elif etype == Schedule.Move:
                ions = info["ions"]                                         
                for ion in ions:
                    self.ion_heating[ion] += self.HEAT_MOVE                

            # --- MERGE ---
            elif etype == Schedule.Merge:
                trap = info["trap"]                                         
                incoming_ions = info["ions"]                                
                existing_ions = replay_traps[trap]                          

                # 1. 更新位置
                new_chain = existing_ions + incoming_ions                   
                replay_traps[trap] = new_chain                              

                # 2. Merge 加热
                for ion in new_chain:
                    self.ion_heating[ion] += self.HEAT_MERGE                

                # 3. 热化 (Thermalization)
                total_heat = sum(self.ion_heating[i] for i in new_chain)    
                avg_heat = total_heat / len(new_chain) if len(new_chain) > 0 else 0   
                for ion in new_chain:
                    self.ion_heating[ion] = avg_heat                        

        # === 3. 最终统计 ===
        t1_decay = math.exp(-self.prog_fin_time / self.T1)                       
        self.final_fidelity = math.exp(log_fidelity) * t1_decay                  
        total_system_heating = sum(self.ion_heating.values())               

        # 打印详细结果 (用于 run.py 观察)
        print(f"Program Finish Time: {self.prog_fin_time} us")
        print("OPCOUNTS Gate:", self.op_count[Schedule.Gate], "Split:", self.op_count[Schedule.Split], 
              "Move:", self.op_count[Schedule.Move], "Merge:", self.op_count[Schedule.Merge])
        
        if len(self.gate_chain_lengths) > 0:                                
            import numpy as np
            lens = np.array(self.gate_chain_lengths)
            print("\n【Two-qubit gate chain length distribution statistics】")
            print(f"  Number of gates : {len(lens)}")
            print(f"  Mean chain length: {np.mean(lens):.2f}")
            print(f"  Median          : {np.median(lens):.1f}")
            print(f"  Max chain length: {np.max(lens)}")
            print(f"  90th percentile : {np.percentile(lens, 90):.1f}")
            print(f"  Gates with length ≥11: {sum(lens >= 11)} ({sum(lens >= 11)/len(lens)*100:.1f}%)")
            print(f"  Gates with length ≥12: {sum(lens >= 12)} ({sum(lens >= 12)/len(lens)*100:.1f}%)")

        print(f"Fidelity: {self.final_fidelity}")                                
        print(f"Total System Heating (quanta): {int(total_system_heating)}")   
        
        if hasattr(self, 'chain_nbar_records') and self.chain_nbar_records:
            records = sorted(self.chain_nbar_records, key=lambda x: x[1], reverse=True)  
            print("\n【Gate information on highest heating chains (top 10)】")
            for i, (trap, avg_n, N, fid) in enumerate(records[:10]):
                print(f"  {i+1}. trap={trap:2d}  avg_n_bar={avg_n:6.1f}  chain length={N:2d}  gate fidelity={fid:.4e}")
            print(f"  Global max avg_n_bar: {records[0][1]:.1f}")

    def analyze_and_return(self):
        """
        运行分析并返回关键指标字典，专门用于 PSO 优化脚本。
        """
        self.move_check() # 填充 self.final_fidelity, self.op_count 等
        
        # 穿梭总次数 = Split + Merge + Move
        shuttle_count = (self.op_count.get(Schedule.Split, 0) + 
                         self.op_count.get(Schedule.Merge, 0) + 
                         self.op_count.get(Schedule.Move, 0))
        
        return {
            'fidelity': self.final_fidelity,
            'total_shuttle': shuttle_count,
            'time': self.prog_fin_time
        }

