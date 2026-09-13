# -*- coding: utf-8 -*-
"""全时域 MPC 执行层：每 10 分钟槽都对剩余日重解 LP（不只是缺口槽）。

与现有滚动执行的区别：
- 现有：仅在 net<0（缺口）时调用 exec_lp 决定放电 vs 紧急；盈余时贪心充电
- MPC：每个槽都重解剩余日 LP（g锁定 + 当前SOC + 最新预测 → 最优充放/弃/紧急），
  从第0槽到第143槽连续决策，放电时机全局最优

计算量：144 LP/天 × 365天 = 52,560 LP ≈ 25分钟（服务器）
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from common import load_all, solve_plan, log, RESULTS, N, DT, ETA, SOC_INIT, DAYS
from q2_saa import solve_saa_plan
from q2_simulate import PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()
E_MAX = 5000 * DT
SOC_MIN, SOC_MAX = 1200.0, 10800.0


def mpc_exec_lp(price_k, load_kw_k, pv_kw_k, g_lock_k, soc, soc_target):
    """MPC 单步：剩余槽 LP。返回本槽 (c, d, emg, spill)。"""
    n = len(g_lock_k)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    nv = 6 * n
    G, C, D_, SP, EM, S = [slice(i * n, (i + 1) * n) for i in range(6)]
    obj = np.zeros(nv)
    obj[EM] = 5.0 * price_k
    obj[C] = 1e-7; obj[D_] = 1e-7
    load_e = np.asarray(load_kw_k) * DT
    pv_e = np.asarray(pv_kw_k) * DT
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(n):
        rows.append(r); cols.append(S.start + t); vals.append(1.0)
        if t > 0:
            rows.append(r); cols.append(S.start + t - 1); vals.append(-1.0)
        else:
            beq.append(soc)
        rows += [r, r]; cols += [C.start + t, D_.start + t]; vals += [-ETA, 1.0 / ETA]
        if t > 0:
            beq.append(0.0)
        r += 1
    for t in range(n):
        rows += [r, r, r, r, r]
        cols += [G.start + t, D_.start + t, EM.start + t, C.start + t, SP.start + t]
        vals += [1.0, 1.0, 1.0, -1.0, -1.0]
        beq.append(load_e[t] - pv_e[t])
        r += 1
    rows.append(r); cols.append(S.start + n - 1); vals.append(1.0)
    beq.append(soc_target)
    r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    b = np.array(beq)
    bounds = [(0, None)] * nv
    for t in range(n):
        bounds[C.start + t] = (0, E_MAX)
        bounds[D_.start + t] = (0, E_MAX)
        bounds[S.start + t] = (SOC_MIN, SOC_MAX)
        bounds[G.start + t] = (g_lock_k[t], g_lock_k[t])
    res = linprog(obj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:
        # 终端不可行 → 放松终端
        A2 = A[:r - 1]; b2 = b[:r - 1]
        res = linprog(obj, A_eq=A2, b_eq=b2, bounds=bounds, method="highs")
        if res.status != 0:
            return 0.0, 0.0, 0.0, 0.0
    x = res.x
    return x[C][0], x[D_][0], x[EM][0], x[SP][0]


def simulate_mpc(lf, pf, price_days, S=10):
    """hybrid2 + SAA + 全时域 MPC 执行。"""
    price = np.broadcast_to(np.asarray(price_days, float), (DAYS, N))
    err_all = (lf - pf) - (LOAD - PV)
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    t0 = time.time()
    for d in range(DAYS):
        soc_start[d] = soc
        errs = err_all[max(0, d - S):d] if d else np.zeros((0, N))
        if len(errs):
            g = solve_saa_plan(price[d], lf[d], pf[d], errs, soc, S=len(errs))
        else:
            g = solve_plan(price[d], lf[d], pf[d], soc0=soc, cyclic=True,
                           validate=False)["g"]
        g_plan[d] = g
        cost_plan[d] = float(price[d] @ g)
        for k in range(N):
            load_k = np.concatenate([[LOAD[d, k]], lf[d, k + 1:]])
            pv_k = np.concatenate([[PV[d, k]], pf[d, k + 1:]])
            ch, dis, em, sp = mpc_exec_lp(price[d][k:], load_k, pv_k, g[k:],
                                           soc, soc_start[d])
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += ETA * ch - dis / ETA
        cost_emg[d] = float(price[d] @ EM[d]) * 5.0
        if d % 50 == 0:
            log(f"  MPC day {d}: {time.time() - t0:.0f}s")
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg, cost_total=cost_plan + cost_emg)
    log(f"  MPC(S={S}): 回测期 {out['cost_total'][REP].sum():,.0f} "
        f"(计划 {out['cost_plan'][REP].sum():,.0f}, 紧急 {out['cost_emg'][REP].sum():,.0f}), "
        f"{time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    r = simulate_mpc(lf, pf, PRICE_A1[None, :], S=10)
    base = 13879046
    tot = float(r["cost_total"][REP].sum())
    print(f"\nMPC vs 滚动执行: {tot:,.0f} vs {base:,.0f} → {tot - base:+,.0f} "
          f"({(tot - base) / base * 100:+.2f}%)")
    np.savez(RESULTS / "q2_mpc_sim.npz",
             **{k: r[k] for k in ["g_plan", "c", "d", "spill", "emg",
                                  "soc_start", "cost_plan", "cost_emg"]})
