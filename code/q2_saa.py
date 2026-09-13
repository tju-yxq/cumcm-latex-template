# -*- coding: utf-8 -*-
"""问题二模型升级：两阶段随机规划（场景法 SAA）。

报童分位数缓冲只利用了净负荷误差的逐槽边缘分位数，且不感知储能的跨时段耦合。
本模块以近 28 天的净负荷预测误差整条曲线为经验场景集，建立两阶段随机规划：
  第一阶段（0:00）：计划购电量 g_t（take-or-pay，所有场景共用）；
  第二阶段（场景 s 展开）：充 c^s、放 d^s、弃 v^s、紧急 e^s 及各场景 SOC 轨迹；
  目标：min Σp·g + (1/S)Σ_s Σ 5p·e^s   （期望总费用）
  约束：各场景 SOC 平衡/功率上限/电量区间/日循环、供需平衡。
该模型是报童缓冲（无储能单时段特例）的严格推广，将风险决策内生化。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from common import (load_all, solve_plan, key_numbers, log, RESULTS,
                    N, DT, E_MAX, SOC_MIN, SOC_MAX, ETA, SOC_INIT, DAYS)
from q2_simulate import simulate, exec_lp, PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()


def solve_saa_plan(price, lf, pf, err_hist, soc0, S=28):
    """两阶段 SAA。err_hist: (n,144) 净负荷误差(预测-实际, kW)，取最近 S 天为场景。
    返回 g (144,)。"""
    S = min(S, len(err_hist))
    if S == 0:                                   # 无历史 → 退化为点预测 LP
        sol = solve_plan(price, lf, pf, soc0=soc0, cyclic=True, validate=False)
        return sol["g"]
    errs = err_hist[-S:]
    n_scn = (lf - pf)[None, :] - errs            # (S,144) 各场景实际净负荷 kW
    nE = n_scn * DT                              # kWh

    NV = N + 5 * N * S                           # g + (c,d,v,e,soc)×S
    G0 = 0
    def C0(s): return N + 5 * N * s
    def D0(s): return N + 5 * N * s + N
    def V0(s): return N + 5 * N * s + 2 * N
    def E0(s): return N + 5 * N * s + 3 * N
    def S0(s): return N + 5 * N * s + 4 * N

    obj = np.zeros(NV)
    obj[:N] = price
    for s in range(S):
        obj[E0(s):E0(s) + N] = 5.0 * price / S
        obj[C0(s):C0(s) + N] = 1e-7              # 微罚吞吐防同时充放退化
        obj[D0(s):D0(s) + N] = 1e-7

    rows, cols, vals, beq = [], [], [], []
    r = 0
    for s in range(S):
        for t in range(N):                       # SOC 平衡
            rows += [r]; cols += [S0(s) + t]; vals += [1.0]
            if t > 0:
                rows += [r]; cols += [S0(s) + t - 1]; vals += [-1.0]
            else:
                beq.append(soc0)
            rows += [r, r]; cols += [C0(s) + t, D0(s) + t]; vals += [-ETA, 1.0 / ETA]
            if t > 0:
                beq.append(0.0)
            r += 1
        rows += [r]; cols += [S0(s) + N - 1]; vals += [1.0]; beq.append(soc0)  # 日循环
        r += 1
        for t in range(N):                       # 供需平衡
            rows += [r]; cols += [G0 + t]; vals += [1.0]
            rows += [r, r, r, r]
            cols += [D0(s) + t, E0(s) + t, C0(s) + t, V0(s) + t]
            vals += [1.0, 1.0, -1.0, -1.0]
            beq.append(nE[s, t])
            r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, NV)).tocsr()
    b = np.array(beq)

    bounds = [(0, None)] * NV
    for s in range(S):
        for t in range(N):
            bounds[C0(s) + t] = (0, E_MAX)
            bounds[D0(s) + t] = (0, E_MAX)
            bounds[S0(s) + t] = (SOC_MIN, SOC_MAX)
    res = linprog(obj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    assert res.status == 0, f"SAA LP 失败: {res.message}"
    return res.x[:N]


def simulate_saa(load_fc, pv_fc, price_days, S=28, exec_mode="rolling"):
    """全年仿真：SAA 计划 + 与问题二相同的执行层。"""
    price = np.broadcast_to(np.asarray(price_days, float), (DAYS, N))
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    err_hist = []
    t0 = time.time()
    for d in range(DAYS):
        lf, pf = load_fc[d], pv_fc[d]
        soc_start[d] = soc
        g = solve_saa_plan(price[d], lf, pf, np.array(err_hist) if err_hist else np.zeros((0, N)),
                           soc, S=S)
        g_plan[d] = g
        cost_plan[d] = float(price[d] @ g)
        net = g + PV[d] * DT - LOAD[d] * DT
        for k in range(N):
            nk = net[k]
            if nk >= 0:
                ch = min(nk, E_MAX, (SOC_MAX - soc) / ETA)
                dis, em, sp = 0.0, 0.0, nk - ch
            else:
                deficit = -nk
                if exec_mode == "rolling" and soc > SOC_MIN + 1e-6:
                    load_k = np.concatenate([[LOAD[d, k]], lf[k + 1:]])
                    pv_k = np.concatenate([[PV[d, k]], pf[k + 1:]])
                    ch, dis, em, sp = exec_lp(price[d][k:], load_k, pv_k, g[k:], soc, soc_start[d])
                else:
                    dis = min(deficit, E_MAX, (soc - SOC_MIN) * ETA)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += ETA * ch - dis / ETA
        cost_emg[d] = float(price[d] @ EM[d]) * 5.0
        err_hist.append((lf - pf) - (LOAD[d] - PV[d]))
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg, cost_total=cost_plan + cost_emg,
               spill_total=float(SP.sum()), emg_total=float(EM.sum()))
    log(f"  SAA仿真(S={S}, {exec_mode}): 回测期总费用 {out['cost_total'][REP].sum():,.0f} 元 "
        f"(计划 {out['cost_plan'][REP].sum():,.0f}, 紧急 {out['cost_emg'][REP].sum():,.0f}), "
        f"{time.time() - t0:.1f}s")
    return out


def main():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    # 1月（过渡期）选场景数 S
    jan = {}
    for S in [10, 14, 28]:
        s = simulate_saa(lf, pf, PRICE_A1[None, :], S=S, exec_mode="greedy")
        jan[S] = float(s["cost_total"][:31].sum())
        print(f"  S={S}: 1月总费用 {jan[S]:,.0f} 元")
    S_star = min(jan, key=jan.get)
    log(f"SAA 选定场景数 S* = {S_star}")

    # 正式回测：SAA × 滚动执行
    sim_saa = simulate_saa(lf, pf, PRICE_A1[None, :], S=S_star, exec_mode="rolling")
    # 对照：报童缓冲主策略（q=0.7 + 滚动，已有结果）
    fc_kn = RESULTS / "key_numbers.json"
    import json
    kn = json.loads(fc_kn.read_text(encoding="utf-8"))
    buf_cost = kn["q2_cost_main"]
    saa_cost = float(sim_saa["cost_total"][REP].sum())
    print(f"\nSAA(S={S_star})+滚动: {saa_cost:,.0f} 元 vs 报童缓冲+滚动: {buf_cost:,.0f} 元 "
          f"→ {'SAA 更优' if saa_cost < buf_cost else '缓冲更优'}，差 {abs(saa_cost - buf_cost):,.0f} 元")

    if saa_cost < buf_cost:
        np.savez(RESULTS / "q2_main_sim.npz",
                 **{k: sim_saa[k] for k in ["g_plan", "c", "d", "spill", "emg",
                                             "soc_start", "cost_plan", "cost_emg"]})
        log("SAA 结果已覆盖 q2_main_sim.npz（成为问题二主策略）")
    key_numbers(q2_saa_S=S_star, q2_saa_cost=round(saa_cost, 0),
                q2_saa_jan={str(k): round(v, 0) for k, v in jan.items()},
                q2_main_is_saa=bool(saa_cost < buf_cost))


if __name__ == "__main__":
    main()
