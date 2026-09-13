# -*- coding: utf-8 -*-
"""全局灵敏度分析：储能效率、紧急购电倍数、SOC 安全区间。

- P1：确定性日循环 LP 对 η / SOC 区间的敏感性（毫秒级）
- P2：主策略（贪心执行）对 η / SOC 区间 / 紧急倍数的一阶敏感性
  紧急倍数 m 下同时重选报童分位数 q=(m-1)/m（理论最优），报告"固定策略"与"重选q"两种口径
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common
import q2_simulate
from q2_simulate import simulate, REP, PRICE_A1
from common import key_numbers, log, RESULTS, FIGURES, N, SOC_INIT

A1, LOAD, PV, PRICE4, FC3 = common.load_all()
price_c = np.roll(A1["price"], 1); load_c = np.roll(A1["load"], 1); pv_c = np.roll(A1["pv"], 1)
fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf, pf = fc["load_fc"], fc["pv_fc"]

# ---- P1：效率与 SOC 区间扫描 ----
def p1_cost(eta=None, smin=None, smax=None):
    if eta is not None: common.ETA = eta
    if smin is not None: common.SOC_MIN = smin
    if smax is not None: common.SOC_MAX = smax
    sol = common.solve_plan(price_c, load_c, pv_c, soc0=SOC_INIT, cyclic=True, validate=False)
    common.ETA, common.SOC_MIN, common.SOC_MAX = 0.9, 1200.0, 10800.0
    assert sol["status"] == 0
    return sol["cost_plan"]

p1_eta = {}
for eta in [0.85, 0.875, 0.9, 0.925, 0.95]:
    p1_eta[eta] = p1_cost(eta=eta)
p1_soc = {}
for smin, smax in [(1200, 10800), (960, 10800), (1440, 10800), (1200, 8640), (1200, 12000)]:
    p1_soc[(smin, smax)] = p1_cost(smin=smin, smax=smax)
print("P1 效率敏感性:", {k: round(v, 2) for k, v in p1_eta.items()})
print("P1 SOC区间敏感性:", {str(k): round(v, 2) for k, v in p1_soc.items()})

# ---- P2：效率与 SOC 区间（贪心执行，策略结构不变）----
def p2_total(eta=None, smin=None, smax=None, q=0.7):
    saved = (common.ETA, common.SOC_MIN, common.SOC_MAX,
             q2_simulate.ETA, q2_simulate.SOC_MIN, q2_simulate.SOC_MAX)
    if eta is not None: common.ETA = q2_simulate.ETA = eta
    if smin is not None: common.SOC_MIN = q2_simulate.SOC_MIN = smin
    if smax is not None: common.SOC_MAX = q2_simulate.SOC_MAX = smax
    s = simulate(lf, pf, PRICE_A1[None, :], buffer_q=q, exec_mode="greedy")
    (common.ETA, common.SOC_MIN, common.SOC_MAX,
     q2_simulate.ETA, q2_simulate.SOC_MIN, q2_simulate.SOC_MAX) = saved
    return float(s["cost_total"][REP].sum())

p2_eta = {eta: p2_total(eta=eta) for eta in [0.85, 0.9, 0.95]}
p2_soc = {(960, 10800): p2_total(smin=960), (1440, 10800): p2_total(smin=1440),
          (1200, 8640): p2_total(smax=8640), (1200, 12000): p2_total(smax=12000)}
print("P2 效率敏感性:", {k: round(v / 1e4, 1) for k, v in p2_eta.items()}, "万元")
print("P2 SOC区间敏感性:", {str(k): round(v / 1e4, 1) for k, v in p2_soc.items()}, "万元")

# ---- P2：紧急倍数 m（固定策略 vs 重选 q=(m-1)/m）----
p2_emerg = {}
for m in [3, 4, 5, 6, 8]:
    q_opt = (m - 1) / m
    s_fix = simulate(lf, pf, PRICE_A1[None, :], buffer_q=0.7, exec_mode="greedy")
    fix_total = float((s_fix["cost_plan"][REP] + s_fix["cost_emg"][REP] / 5.0 * m).sum())
    s_re = simulate(lf, pf, PRICE_A1[None, :], buffer_q=q_opt, exec_mode="greedy")
    re_total = float((s_re["cost_plan"][REP] + s_re["cost_emg"][REP] / 5.0 * m).sum())
    p2_emerg[m] = (fix_total, re_total)
    print(f"  紧急倍数 m={m}: 固定q=0.7 {fix_total/1e4:.1f} 万, 重选q={q_opt:.2f} {re_total/1e4:.1f} 万")

key_numbers(
    sens_p1_eta={str(k): round(v, 2) for k, v in p1_eta.items()},
    sens_p1_soc={str(k): round(v, 2) for k, v in p1_soc.items()},
    sens_p2_eta={str(k): round(v, 0) for k, v in p2_eta.items()},
    sens_p2_soc={str(k): round(v, 0) for k, v in p2_soc.items()},
    sens_p2_emerg={str(m): [round(a, 0), round(b, 0)] for m, (a, b) in p2_emerg.items()},
)

# ---- 图：三联灵敏度 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
ax = axes[0]
ks = sorted(p1_eta); ax.plot([k * 100 for k in ks], [p1_eta[k] for k in ks], "o-", color="#2c3e50")
ax.axvline(90, color="#c0392b", ls="--", lw=1)
ax.set_xlabel("充放电效率 η (%)"); ax.set_ylabel("问题一日购电费（元）")
ax.set_title("(a) 效率敏感性（问题一）")
ax = axes[1]
ks = sorted(p2_eta); ax.plot([k * 100 for k in ks], [p2_eta[k] / 1e4 for k in ks], "o-", color="#2c3e50")
ax.axvline(90, color="#c0392b", ls="--", lw=1)
ax.set_xlabel("充放电效率 η (%)"); ax.set_ylabel("问题二总费用（万元）")
ax.set_title("(b) 效率敏感性（问题二）")
ax = axes[2]
ms = sorted(p2_emerg)
ax.plot(ms, [p2_emerg[m][0] / 1e4 for m in ms], "s--", color="#7f8c8d", label="固定 q=0.7")
ax.plot(ms, [p2_emerg[m][1] / 1e4 for m in ms], "o-", color="#c0392b", label="按理论重选 q")
ax.axvline(5, color="#888", ls=":", lw=1)
ax.set_xlabel("紧急购电价格倍数 m"); ax.set_ylabel("问题二总费用（万元）")
ax.set_title("(c) 紧急倍数敏感性（问题二）"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(FIGURES / "fig_sens.png", dpi=300); plt.close(fig)
log("灵敏度分析与图已输出")
