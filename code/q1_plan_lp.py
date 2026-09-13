# -*- coding: utf-8 -*-
"""问题1：电价与负载逐日相同（附件1典型日）+ 光伏预测已知 → 确定性日循环 LP。

模型（时钟日 144 槽 j=0..143，槽 j = [j*10,(j+1)*10) 分钟）：
  min  Σ p_j g_j
  s.t. s_j = s_{j-1} + 0.9 c_j - d_j/0.9        (SOC 平衡, s_{-1}=6000)
       1200 ≤ s_j ≤ 10800,  0 ≤ c_j,d_j ≤ 833.33
       g_j + pv_j·Δt + d_j = load_j·Δt + c_j + spill_j   (供需平衡, 弃电 spill ≥ 0)
       s_143 = 6000                                (0:00 与 24:00 储电量相同)
附件1 文件列序(0:10..0:00+1) → 时钟日: clock = roll(file, 1)；
结果文件按计划日文件列序填写: file = roll(clock, -1)。
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import (load_all, solve_plan, to_file_order, key_numbers, log,
                    N, DT, SOC_INIT, RESULTS, FIGURES)

a1, load, pv, price4, fc3 = load_all()

# ---- 时钟日数据（典型日，每一天相同；槽 j 的数据 = 文件列 (j-1)%144）----
price_c = np.roll(a1["price"], 1)
load_c = np.roll(a1["load"], 1)
pv_c = np.roll(a1["pv"], 1)

sol = solve_plan(price_c, load_c, pv_c, soc0=SOC_INIT, cyclic=True)
assert sol["status"] == 0
g, c, d, s, spill = sol["g"], sol["c"], sol["d"], sol["s"], sol["spill"]
cost, total_g = sol["cost_plan"], float(g.sum())
log(f"P1 求解成功: 全天购电量 {total_g:.2f} kWh, 购电费 {cost:.2f} 元")

# ---- 无储能基准 ----
base_g = np.maximum(load_c * DT - pv_c * DT, 0.0)
base_cost = float(price_c @ base_g)
base_total = float(base_g.sum())
save_abs, save_pct = base_cost - cost, (base_cost - cost) / base_cost * 100
log(f"无储能基准: 购电量 {base_total:.2f} kWh, 购电费 {base_cost:.2f} 元; "
    f"储能节省 {save_abs:.2f} 元 ({save_pct:.2f}%)")

# ---- 储能日循环电量水平灵敏度（s0 自由时应选哪个水平）----
levels = np.arange(1200, 10801, 600)
level_cost = []
for lv in levels:
    r = solve_plan(price_c, load_c, pv_c, soc0=float(lv), cyclic=True)
    level_cost.append(r["cost_plan"] if r["status"] == 0 else np.nan)
level_cost = np.array(level_cost)
best_i = int(np.nanargmin(level_cost))
log(f"灵敏度: 最优日循环水平 {levels[best_i]} kWh, 费用 {level_cost[best_i]:.2f} 元 "
    f"(6000kWh 时 {cost:.2f} 元, 差 {cost-level_cost[best_i]:.2f} 元)")

# ---- 表1：指定时段购电量 ----
t1_idx = [60, 72, 84, 96, 108, 120]   # 10:00,12:00,...,20:00 槽
t1_labels = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
             "16:00-16:10", "18:00-18:10", "20:00-20:10"]
t1_vals = [float(g[j]) for j in t1_idx]

# ---- 表2：4小时块充放电 + 0/24 点储电量 ----
blocks = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]
t2 = [(lab, float(c[a:b].sum()), float(d[a:b].sum())) for lab, a, b in blocks]

# ---- 写 result1.xlsx（保持模板结构）----
from openpyxl import load_workbook
tpl = r"C:\Users\26517\Documents\CUMCM\C题\附件\附件5\result1.xlsx"
wb = load_workbook(tpl)
ws = wb["计划购电量"]
g_file = to_file_order(g)              # 计划日文件列序
assert ws.max_row == N + 1, f"模板行数 {ws.max_row} != {N+1}"
assert str(ws.cell(row=2, column=1).value).startswith("0:10")
assert str(ws.cell(row=N + 1, column=1).value).startswith("0:00+1")
for i in range(N):
    ws.cell(row=i + 2, column=2, value=round(float(g_file[i]), 4))
ws2 = wb["充放电量"]
for i, (lab, cc, dd) in enumerate(t2):
    ws2.cell(row=i + 2, column=2, value=round(cc, 4))
    ws2.cell(row=i + 2, column=3, value=round(dd, 4))
ws2.cell(row=2, column=5, value=round(SOC_INIT, 4))      # 0:00 储电量
ws2.cell(row=3, column=5, value=round(float(s[-1]), 4))  # 24:00 储电量
out_path = RESULTS / "result1.xlsx"
wb.save(out_path)
log(f"已写出 {out_path}")

# ---- 图1：策略图（电价+购电/充放电 + SOC）----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

tmin = np.arange(N) * 10 / 60.0  # 小时
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.6]})
ax = axes[0]
ax.plot(tmin, price_c, color="#888888", lw=1.4, label="电价（右轴）")
ax.set_ylabel("电价 (元/kWh)", color="#888888")
ax.tick_params(axis="y", labelcolor="#888888")
ax2 = ax.twinx()
ax2.step(tmin, g, where="post", color="#c0392b", lw=1.6, label="购电量")
ax2.bar(tmin, c, width=10/60, align="edge", color="#2471a3", alpha=0.55, label="充电量")
ax2.bar(tmin, -d, width=10/60, align="edge", color="#1e8449", alpha=0.55, label="放电量")
ax2.set_ylabel("电量 (kWh / 10min)")
ax2.axhline(0, color="k", lw=0.6)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=4, framealpha=0.9)
ax.set_title("问题一：最优购电与储能充放电策略（典型日）")
axs = axes[1]
axs.plot(tmin, s, color="#6c3483", lw=1.8)
axs.axhline(1200, color="r", ls="--", lw=0.8); axs.axhline(10800, color="r", ls="--", lw=0.8)
axs.set_ylabel("储电量 (kWh)"); axs.set_xlabel("时刻 (h)")
axs.set_ylim(0, 12000)
fig.tight_layout()
fig.savefig(FIGURES / "fig_q1_strategy.png", dpi=300)
plt.close(fig)
log("已输出 figures/fig_q1_strategy.png")

# ---- 关键数值入库 ----
key_numbers(
    q1_total_purchase=round(total_g, 2), q1_total_cost=round(cost, 2),
    q1_baseline_cost=round(base_cost, 2), q1_save_abs=round(save_abs, 2),
    q1_save_pct=round(save_pct, 2),
    q1_table1={l: round(v, 2) for l, v in zip(t1_labels, t1_vals)},
    q1_table2={lab: [round(cc, 2), round(dd, 2)] for lab, cc, dd in t2},
    q1_soc_0=round(SOC_INIT, 2), q1_soc_24=round(float(s[-1]), 2),
    q1_best_level=float(levels[best_i]),
    q1_best_level_cost=round(float(level_cost[best_i]), 2),
    q1_spill_total=round(float(spill.sum()), 2),
    q1_charge_total=round(float(c.sum()), 2), q1_discharge_total=round(float(d.sum()), 2),
)
log("关键数值已写入 key_numbers.json")

# 控制台摘要
print("\n===== 表1 微网在指定时间段的购电量 =====")
for l, v in zip(t1_labels, t1_vals):
    print(f"  {l}: {v:.2f} kWh")
print(f"  全天购电量: {total_g:.2f} kWh, 全天购电费: {cost:.2f} 元")
print("===== 表2 储能设备充放电量 =====")
for lab, cc, dd in t2:
    print(f"  {lab}: 充 {cc:.2f}, 放 {dd:.2f} kWh")
print(f"  0:00 储电量 {SOC_INIT:.0f}, 24:00 储电量 {s[-1]:.2f} kWh")
print(f"  弃电量 {spill.sum():.2f} kWh；无储能基准费 {base_cost:.2f} 元")
