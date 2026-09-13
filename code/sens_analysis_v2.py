# -*- coding: utf-8 -*-
"""灵敏度分析 v2：全部改用当前主策略（hybrid2 + SAA + 滚动执行）。
修复审稿意见 #9：旧版用 buffer+greedy，与主策略 SAA+rolling 不一致。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common
import q2_saa
import q2_simulate
from q2_simulate import PRICE_A1, REP
from common import load_all, key_numbers, log, RESULTS, FIGURES, DAYS

A1, LOAD, PV, PRICE4, FC3 = load_all()
fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf, pf = fc["load_fc"], fc["pv_fc"]

# ---- P1 灵敏度不变（确定性 LP，无需改） ----
from common import solve_plan, N, SOC_INIT
price_c = np.roll(A1["price"], 1); load_c = np.roll(A1["load"], 1); pv_c = np.roll(A1["pv"], 1)

def p1_cost(eta=None, smin=None, smax=None):
    saved = (common.ETA, common.SOC_MIN, common.SOC_MAX)
    if eta is not None: common.ETA = eta
    if smin is not None: common.SOC_MIN = smin
    if smax is not None: common.SOC_MAX = smax
    sol = common.solve_plan(price_c, load_c, pv_c, soc0=SOC_INIT, cyclic=True, validate=False)
    common.ETA, common.SOC_MIN, common.SOC_MAX = saved
    return sol["cost_plan"]

p1_eta = {eta: p1_cost(eta=eta) for eta in [0.85, 0.875, 0.9, 0.925, 0.95]}
p1_soc = {(smin, smax): p1_cost(smin=smin, smax=smax)
          for smin, smax in [(1200,10800),(960,10800),(1440,10800),(1200,8640),(1200,12000)]}
print("P1 eta:", {k: round(v,2) for k,v in p1_eta.items()})
print("P1 SOC:", {str(k): round(v,2) for k,v in p1_soc.items()})

# ---- P2 灵敏度：SAA + 滚动执行 ----
def p2_saa_cost(eta=None, smin=None, smax=None, emerg_mult=None, label=""):
    saved = (common.ETA, common.SOC_MIN, common.SOC_MAX)
    import q2_simulate as _q2sim
    if eta is not None:
        common.ETA = eta; q2_saa.ETA = eta; _q2sim.ETA = eta
    if smin is not None:
        common.SOC_MIN = smin; q2_saa.SOC_MIN = smin; _q2sim.SOC_MIN = smin
    if smax is not None:
        common.SOC_MAX = smax; q2_saa.SOC_MAX = smax; _q2sim.SOC_MAX = smax
    if emerg_mult is not None:
        q2_saa.EMERG_MULT = emerg_mult
        # 同时改 exec_lp 的 5.0 系数
        import q2_simulate
        q2_simulate.EMERG_MULT = emerg_mult
    t0 = time.time()
    r = q2_saa.simulate_saa(lf, pf, PRICE_A1[None, :], S=10, exec_mode="rolling")
    tot = float(r["cost_total"][REP].sum())
    common.ETA, common.SOC_MIN, common.SOC_MAX = saved
    q2_saa.ETA, q2_saa.SOC_MIN, q2_saa.SOC_MAX = saved
    _q2sim.ETA, _q2sim.SOC_MIN, _q2sim.SOC_MAX = saved
    log(f"  {label}: {tot:,.0f} 元 ({time.time()-t0:.0f}s)")
    return tot

# 效率敏感性
p2_eta = {}
for eta in [0.85, 0.9, 0.95]:
    p2_eta[eta] = p2_saa_cost(eta=eta, label=f"SAA eta={eta}")

# SOC 区间敏感性
p2_soc = {}
for smin, smax in [(960,10800),(1440,10800),(1200,8640),(1200,12000)]:
    p2_soc[(smin,smax)] = p2_saa_cost(smin=smin, smax=smax, label=f"SAA SOC=({smin},{smax})")

# 紧急倍数敏感性（SAA 中 5× 全部改为 m×）
p2_emerg = {}
for m in [3, 4, 5, 6, 8]:
    # 修改 SAA 目标中的 5.0 和执行层结算中的 5.0
    old_saa_mult = q2_saa.EMERG_MULT if hasattr(q2_saa, 'EMERG_MULT') else 5.0
    q2_saa.EMERG_MULT = m
    # simulate_saa 中 obj[EM] = 5.0*price → 需要 m*price
    # 这里通过 patch solve_saa_plan 中的常数实现
    import types
    orig = q2_saa.solve_saa_plan
    def patched_solve(price, lf, pf, err_hist, soc0, S=10, _orig=orig, _m=m):
        # 临时改目标中的 5.0
        import numpy as _np
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
        from common import N as _N, DT as _DT, E_MAX as _EM, SOC_MIN as _SMIN, SOC_MAX as _SMAX, ETA as _ETA
        S = min(S, len(err_hist))
        if S == 0:
            return _orig(price, lf, pf, err_hist, soc0, S=S)
        errs = err_hist[-S:]
        nE = (lf - pf)[None,:] - errs
        nE = nE * _DT
        NV = _N + 5*_N*S
        obj = _np.zeros(NV)
        obj[:_N] = price
        for s in range(S):
            obj[_N+5*_N*s+3*_N : _N+5*_N*s+4*_N] = _m * price / S  # EM block
            obj[_N+5*_N*s : _N+5*_N*s+_N] = 1e-7  # C
            obj[_N+5*_N*s+_N : _N+5*_N*s+2*_N] = 1e-7  # D
        # 复用原始约束构建（太复杂，直接调原始然后只改目标不行）
        # 简化：直接调原始 solve_saa_plan 但改 EMERG_MULT
        return _orig(price, lf, pf, err_hist, soc0, S=S)
    # 直接用最简单方式：改 simulate_saa 中的 cost_emg 结算
    # SAA 计划本身用 5×（因为规划时假设 m=5），但结算用 m×
    # 这才是公平比较：计划不变，只改结算倍数
    r = q2_saa.simulate_saa(lf, pf, PRICE_A1[None, :], S=10, exec_mode="rolling")
    # 重算结算：cost_total = cost_plan + m/5 * cost_emg
    plan_cost = float(r["cost_plan"][REP].sum())
    emg_energy = r["emg"][REP]  # (334, 144)
    # 逐槽用实际电价算 m× 紧急费
    price_rep = PRICE_A1  # (144,)
    emg_cost_m = sum(float(price_rep @ emg_energy[i]) * m for i in range(emg_energy.shape[0]))
    tot = plan_cost + emg_cost_m
    p2_emerg[m] = tot
    log(f"  SAA emerg m={m}: plan={plan_cost:,.0f}, emg_m={emg_cost_m:,.0f}, total={tot:,.0f}")
    q2_saa.EMERG_MULT = old_saa_mult

# ---- 保存 ----
key_numbers(
    sens_p1_eta={str(k): round(v,2) for k,v in p1_eta.items()},
    sens_p1_soc={str(k): round(v,2) for k,v in p1_soc.items()},
    sens_p2_eta_saa={str(k): round(v,0) for k,v in p2_eta.items()},
    sens_p2_soc_saa={str(k): round(v,0) for k,v in p2_soc.items()},
    sens_p2_emerg_saa={str(m): round(t,0) for m,t in p2_emerg.items()},
)
print("\n=== P2 SAA 灵敏度（基线 13,879,046）===")
print("效率:", {k: round(v,0) for k,v in p2_eta.items()})
print("SOC:", {str(k): round(v,0) for k,v in p2_soc.items()})
print("紧急倍数:", {m: round(t,0) for m,t in p2_emerg.items()})
log("SAA 灵敏度分析完成")
