import subprocess as sp

PATH = "./programs"

# 选择电路
PROG = ["BV32", "GHZ32", "QFT32"]

# 选择架构与容量
MACHINE = {"G2x2": "12", "G2x3": "8"}

# 映射方式
mapper = "SABRE"

# 重排序方式
reorder = "Naive"

# 调度器组合（可自由增删）
SCHEDULERS = [
    ("MUSS", "V2"),   # 论文版本
    ("MUSS", "V3"),   # 创新版本
    ("MUSS", "V4"),   # 创新版本调度基础上的降低加热版本。
    # ("EJF",  ""),     # 如果想加 EJF，也可以扩展
]

for prog in PROG:
    for machine, ions in MACHINE.items():
        for family, version in SCHEDULERS:
            # 日志文件名带上版本信息，便于区分
            log_name = f"./output/{prog}_{machine}_{ions}_{mapper}_{family}_{version}.log"
            
            args = [
                "python", "run.py",
                f"{PATH}/{prog}.qasm",
                machine,
                ions,
                mapper,
                reorder,
                "1", "0", "0",          # serial_trap, serial_comm, serial_all
                "FM", "GateSwap"
            ]
            
            # 只在需要 MUSS 家族时才传版本
            if family == "MUSS":
                args.extend([family, version])
            else:
                args.append(family)     # EJF 等只需要 family

            print(f"Running: {' '.join(args)} → {log_name}")
            
            with open(log_name, "w", encoding="utf-8") as f:
                sp.call(args, stdout=f)
