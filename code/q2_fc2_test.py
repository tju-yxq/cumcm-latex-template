# -*- coding: utf-8 -*-
"""hybrid2 预测器（负载=加权同星期几3周，光伏=weighted7）入缓存 + SAA 成本验证。

选择依据（防泄漏）：
- 光伏 weighted7：1 月窗口（82.8 < mean7 84.2）与全年均最优——协议内干净采纳；
- 负载近期加权：先验选择（非平稳序列的近期遗忘处理）；1 月各窗长 MAE 差 <4 kW
  （240.1-242.6）不敏感，取 3 周为钝感区间中值；全年敏感性 2/3/4/5 周 =
  167.4/169.1/173.9/180.8，加权 vs 不加权（188.9）为主要增益来源。
"""
import sys, io, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy import stats
from common import load_all, DAYS, N, RESULTS
from q2_saa import simulate_saa
from q2_simulate import PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()

# ---- 构造 hybrid2 ----
def load_fc2(d):
    ix = [j for j in range(d - 7, d - 22, -7) if j >= 0]      # d-7, d-14, d-21
    if not ix:
        return LOAD[max(0, d - 1)]
    w = np.arange(len(ix), 0, -1, dtype=float)                 # 3:2:1
    return w @ LOAD[ix] / w.sum()

def pv_fc2(d):
    ix = np.arange(max(0, d - 7), d)
    if len(ix) == 0:
        return PV[max(0, d - 1)]
    w = np.arange(1, len(ix) + 1, dtype=float)
    return w @ PV[ix] / w.sum()

lf2 = np.array([load_fc2(d) for d in range(DAYS)])
pf2 = np.array([pv_fc2(d) for d in range(DAYS)])
act_net = LOAD - PV

# 入缓存
cache = dict(np.load(Path(__file__).parent / "cache" / "forecast_p2.npz"))
cache["hybrid2_load"], cache["hybrid2_pv"] = lf2, pf2
np.savez(Path(__file__).parent / "cache" / "forecast_p2.npz", **cache)
print("hybrid2 已写入缓存")

# 精度 + 显著性
fc_old = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf1, pf1 = fc_old["load_fc"], fc_old["pv_fc"]
d_old = np.abs((lf1 - pf1) - act_net).mean(axis=1)[REP]
d_new = np.abs((lf2 - pf2) - act_net).mean(axis=1)[REP]
w = stats.wilcoxon(d_old, d_new, alternative="greater")
print(f"净负荷MAE: hybrid {d_old.mean():.1f} → hybrid2 {d_new.mean():.1f}  "
      f"Wilcoxon p={w.pvalue:.2e}")

# ---- SAA 成本验证 ----
r = simulate_saa(lf2, pf2, PRICE_A1[None, :], S=10, exec_mode="rolling")
tot = float(r["cost_total"][REP].sum())
base = 13938354.0
print(f"\nhybrid2 + SAA(S=10) + 滚动: {tot:,.0f} 元  vs 当前主策略 {tot - base:+,.0f} "
      f"({(tot - base) / base * 100:+.2f}%)")
print(f"Ma-ya6 门槛 13,834,487: {'✓ 跨过' if tot < 13834487 else '✗ 未过'}")
np.savez(RESULTS / "q2_hybrid2_sim.npz",
         **{k: r[k] for k in ["g_plan", "c", "d", "spill", "emg",
                              "soc_start", "cost_plan", "cost_emg"]})
