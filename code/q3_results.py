# -*- coding: utf-8 -*-
"""问题3 结果输出：result3.xlsx、四指定日明细、论文图表。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from common import load_all, key_numbers, log, RESULTS, FIGURES, N, DT, DAYS
from q2_simulate import PRICE_A1, REP
from xlsx_writer import (write_plan_sheet, write_battery_sheet, write_emergency_sheet,
                         spec_date_detail, soc_boundaries)

A1, LOAD, PV, PRICE4, FC3 = load_all()
sim = np.load(RESULTS / "q3_main_sim.npz")
G, Afin, C, D_, EM = sim["g_plan"], sim["a_final"], sim["c"], sim["d"], sim["emg"]
soc_start, cost_settle, cost_emg = sim["soc_start"], sim["cost_settle"], sim["cost_emg"]

# ---- result3.xlsx：计划 / 调整 / 充放电 / 紧急 ----
tpl = r"C:\Users\26517\Documents\CUMCM\C题\附件\附件5\result3.xlsx"
wb = load_workbook(tpl)
# 计划购电量 sheet 的全天购电费 = 0:00 计划的 p·g
plan_cost = np.array([float(PRICE_A1 @ G[d]) for d in range(DAYS)])
write_plan_sheet(wb["计划购电量"], G, plan_cost)
# 调整购电量 sheet：最终生效量 a_final + 偏差结算费（口径A）
write_plan_sheet(wb["调整购电量"], Afin, cost_settle)
soc_0, soc_24 = soc_boundaries(C, D_, soc_start)
write_battery_sheet(wb["充放电量"], C, D_, soc_0, soc_24)
n_emg_days = write_emergency_sheet(wb["紧急购电量"], EM)
out = RESULTS / "result3.xlsx"
wb.save(out)
log(f"result3.xlsx 已写出（{n_emg_days} 个紧急购电日）")

# ---- 四指定日明细 ----
def cost_by_day(D):
    return dict(total_g=round(float(Afin[D].sum()), 2),
                cost_settle=round(float(cost_settle[D]), 2),
                cost_emg=round(float(cost_emg[D]), 2),
                cost_total=round(float(cost_settle[D] + cost_emg[D]), 2),
                adj_qty=round(float(np.abs(Afin[D] - G[D]).sum()), 2))

detail = spec_date_detail(G, Afin, C, D_, EM, soc_start, cost_by_day)
for s, v in detail.items():
    print(f"{s}: 最终购电 {v['total_g']:.0f} kWh, 结算费 {v['cost_settle']:.0f}, "
          f"紧急 {v['cost_emg']:.0f}, 调整量 {v['adj_qty']:.0f} kWh, "
          f"SOC {v['soc_0']:.0f}->{v['soc_24']:.0f}, 紧急时段 {len(v['t3_emergency'])}")
key_numbers(q3_spec_dates=detail)

# ---- 图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

kn = json.loads((RESULTS / "key_numbers.json").read_text(encoding="utf-8"))
# 图A：调整策略对比
names = ["不调整", "仅6时", "仅12时", "仅18时", "全调整\n(主策略)"]
vals = [kn["q3_cost_noAdj"], kn["q3_cost_adj6"], kn["q3_cost_adj12"],
        kn["q3_cost_adj18"], kn["q3_cost_main"]]
fig, ax = plt.subplots(figsize=(8.2, 4.4))
bars = ax.bar(np.arange(5), np.array(vals) / 1e4, 0.56,
              color=["#95a5a6", "#7fb3d5", "#7fb3d5", "#7fb3d5", "#c0392b"])
for i, v in enumerate(vals):
    ax.text(i, v / 1e4 + 1.2, f"{v/1e4:.1f}", ha="center", fontsize=10)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel("回测期总费用（万元）")
ax.set_title("问题三：引入其他时刻预报的调整价值")
fig.tight_layout(); fig.savefig(FIGURES / "fig_q3_adj_cmp.png", dpi=300); plt.close(fig)

# 图B：典型调整日（|a-g| 最大的日期）：计划 vs 调整 vs 光伏实际
adj_qty = np.abs(Afin - G).sum(axis=1); adj_qty[:31] = 0
Dmax = int(np.argmax(adj_qty))
lab_time = [(k + 1) * 10 / 60 for k in range(N)]
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.6]})
ax = axes[0]
ax.plot(lab_time, PRICE_A1, color="#888888", lw=1.2, label="电价（右轴）")
ax.set_ylabel("电价（元/kWh）", color="#888888"); ax.tick_params(axis="y", labelcolor="#888888")
ax2 = ax.twinx()
ax2.step(lab_time, G[Dmax], where="post", color="#c0392b", lw=1.5, label="0:00 计划购电")
ax2.step(lab_time, Afin[Dmax], where="post", color="#8e44ad", lw=1.5, ls="--", label="调整后购电")
ax2.plot(lab_time, PV[Dmax] * DT, color="#f39c12", lw=1.2, label="光伏实际（×Δt）")
for T, lab in [(6, "6时预报"), (12, "12时预报"), (18, "18时预报")]:
    ax2.axvline(T, color="#555", ls=":", lw=0.9)
ax2.axhline(0, color="k", lw=0.5); ax2.set_ylabel("电量（kWh/10min）")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, ncol=4, framealpha=0.9)
ax.set_title(f"问题三：滚动调整示例（{pd.Timestamp('2025-01-01') + pd.Timedelta(days=Dmax):%Y-%m-%d}，"
             f"调整总量 {adj_qty[Dmax]:.0f} kWh）")
soc_tr = soc_start[Dmax] + np.cumsum(0.9 * C[Dmax] - D_[Dmax] / 0.9)
axes[1].plot(lab_time, soc_tr, color="#6c3483", lw=1.8)
axes[1].axhline(1200, color="r", ls="--", lw=0.8); axes[1].axhline(10800, color="r", ls="--", lw=0.8)
axes[1].set_ylabel("储电量（kWh）"); axes[1].set_xlabel("时刻（h）"); axes[1].set_ylim(0, 12000)
fig.tight_layout(); fig.savefig(FIGURES / "fig_q3_adj_day.png", dpi=300); plt.close(fig)
log(f"图已输出：fig_q3_adj_cmp / fig_q3_adj_day({pd.Timestamp('2025-01-01') + pd.Timedelta(days=Dmax):%m-%d})")
