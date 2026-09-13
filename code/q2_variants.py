# -*- coding: utf-8 -*-
"""反超实验模块：
E1 — SAA 场景按负荷日型分组选择（周五/周六为低需求组，仅用1月暖机期数据判定，无前视）
E2 — 自适应分位数变体池计划层（6 预测器 × q80/85/90/95，逐日按 trailing pinball loss 选择）

背景：周五/周六日均净负荷 ≈950 kW，其余五天 ≈2860 kW（3 倍差）。原 SAA 取"近10天混合"
场景，工作日计划混入低负荷日 → 尾部风险低估 → 紧急购电费偏高。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from common import load_all, solve_plan, log, RESULTS, N, DT, DAYS
from q2_saa import solve_saa_plan
from q2_simulate import exec_lp, PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()
DATES = pd.date_range("2025-01-01", periods=DAYS)
DOW = DATES.dayofweek.to_numpy()
# 低需求组：仅用 1 月暖机期数据判定（周五/周六，实测为其余日 0.48 倍）
LOW_GROUP = np.zeros(7, dtype=bool)
LOW_GROUP[4] = LOW_GROUP[5] = True
GRP = LOW_GROUP[DOW]                      # (365,) bool：True=低需求日


def scenario_errors_days(err_all, d, rule, S=10, lookback=21):
    """按规则选场景日，返回这些日的净负荷误差(预测-实际)矩阵 (n,144)。"""
    if d == 0:
        return np.zeros((0, N))
    if rule == "recent":
        days = list(range(max(0, d - S), d))
    elif rule == "weighted":
        # 近14天池，同周几日的误差重复计入（保留混合池的隐性对冲 + 校准同型日）
        pool = list(range(max(0, d - 14), d))
        rows = [j for j in pool] + [j for j in pool if DOW[j] == DOW[d]]
        return err_all[rows]
    elif rule == "mix10sw2":
        # 近10天 + 最近2个同周几日
        days = list(range(max(0, d - 10), d))
        sw = [j for j in range(d) if DOW[j] == DOW[d]][-2:]
        return err_all[days + sw]
    elif rule == "group":
        cand = [j for j in range(max(0, d - lookback), d) if GRP[j] == GRP[d]]
        days = cand[-S:]
        if len(days) < 3:
            days = list(range(max(0, d - 10), d))
    elif rule == "weekday":
        cand = [j for j in range(d) if DOW[j] == DOW[d]]
        days = cand[-S:]
        if len(days) < 3:
            days = list(range(max(0, d - 10), d))
    else:
        raise ValueError(rule)
    return err_all[days]


# ---------------------------------------------------------------- E1
def simulate_saa_group(load_fc, pv_fc, price_days, rule="group", S=10,
                       exec_mode="rolling"):
    """E1：SAA + 分组场景选择。"""
    price = np.broadcast_to(np.asarray(price_days, float), (DAYS, N))
    err_all = (load_fc - pv_fc) - (LOAD - PV)          # (365,144) 预测-实际
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = 6000.0
    t0 = time.time()
    for d in range(DAYS):
        lf, pf = load_fc[d], pv_fc[d]
        soc_start[d] = soc
        errs = scenario_errors_days(err_all, d, rule, S)
        g = solve_saa_plan(price[d], lf, pf, errs, soc, S=len(errs)) \
            if len(errs) else solve_plan(price[d], lf, pf, soc0=soc, cyclic=True,
                                         validate=False)["g"]
        g_plan[d] = g
        cost_plan[d] = float(price[d] @ g)
        net = g + PV[d] * DT - LOAD[d] * DT
        for k in range(N):
            nk = net[k]
            if nk >= 0:
                ch = min(nk, 5000 * DT, (10800 - soc) / 0.9)
                dis, em, sp = 0.0, 0.0, nk - ch
            else:
                deficit = -nk
                if exec_mode == "rolling" and soc > 1200 + 1e-6:
                    load_k = np.concatenate([[LOAD[d, k]], lf[k + 1:]])
                    pv_k = np.concatenate([[PV[d, k]], pf[k + 1:]])
                    ch, dis, em, sp = exec_lp(price[d][k:], load_k, pv_k, g[k:],
                                               soc, soc_start[d])
                else:
                    dis = min(deficit, 5000 * DT, (soc - 1200) * 0.9)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += 0.9 * ch - dis / 0.9
        cost_emg[d] = float(price[d] @ EM[d]) * 5.0
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg,
               cost_total=cost_plan + cost_emg)
    log(f"  E1[{rule},S={S}]: 回测期总费用 {out['cost_total'][REP].sum():,.0f} 元 "
        f"(计划 {out['cost_plan'][REP].sum():,.0f}, 紧急 {out['cost_emg'][REP].sum():,.0f}), "
        f"{time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------- E2
def _predictors():
    """6 种预测器 → (load_fc, pv_fc) 各 (365,144)。"""
    out = {}
    for name, K, ew in [("persistence", 1, False), ("mean7", 7, False),
                        ("mean14", 14, False), ("mean30", 30, False),
                        ("wmean14", 14, True), ("wmean30", 30, True)]:
        lf = np.zeros((DAYS, N)); pf = np.zeros((DAYS, N))
        for d in range(1, DAYS):
            idx = np.arange(max(0, d - K), d)
            if ew:
                w = np.exp(-(d - 1 - idx) / 7.0); w /= w.sum()
                lf[d] = w @ LOAD[idx]; pf[d] = w @ PV[idx]
            else:
                lf[d] = LOAD[idx].mean(axis=0); pf[d] = PV[idx].mean(axis=0)
        out[name] = (lf, pf)
    return out


def simulate_quantile_pool(price_days, q_mode="proxy", qs=(0.80, 0.85, 0.90, 0.95),
                           K_sel=14, exec_mode="rolling"):
    """E2：分位数变体池计划层。q_mode: "proxy"=逐日按 trailing newsvendor 成本代理
    选 (预测器, q)；float = 固定该 q（预测器仍按同口径成本代理选）。
    成本代理（逐槽、无储能近似）：cost = p·Q + 5p·(A−Q)⁺，其中 Q=分位数预测，A=实际净负荷。"""
    price = np.broadcast_to(np.asarray(price_days, float), (DAYS, N))
    preds = _predictors()
    names = list(preds)
    act_net = LOAD - PV
    err = {p: (preds[p][0] - preds[p][1]) - act_net for p in names}
    price0 = price[31]                                  # 代表性电价曲线（代理用）
    # 分位数预测 Q[p,q][d] 与代理成本
    Q = {(p, q): np.full((DAYS, N), np.nan) for p in names for q in qs}
    cost_proxy = {(p, q): np.full(DAYS, np.inf) for p in names for q in qs}
    for p in names:
        for q in qs:
            for d in range(1, DAYS):
                h = err[p][max(0, d - 28):d]
                Q[(p, q)][d] = (preds[p][0][d] - preds[p][1][d]) \
                    - np.quantile(h, 1 - q, axis=0)
                A = act_net[d] * DT                     # kWh
                Qd = Q[(p, q)][d] * DT
                cost_proxy[(p, q)][d] = float(
                    (price0 * Qd).sum() + (5 * price0 * np.maximum(A - Qd, 0)).sum())
    # 逐日选择
    sel = [None] * DAYS
    cand_qs = qs if q_mode == "proxy" else (q_mode,)
    for d in range(DAYS):
        if d < 15:
            sel[d] = ("mean7", cand_qs[len(cand_qs) // 2])
            continue
        js = list(range(max(1, d - K_sel), d))
        best, best_c = None, np.inf
        for p in names:
            for q in cand_qs:
                c = np.mean([cost_proxy[(p, q)][j] for j in js])
                if c < best_c:
                    best, best_c = (p, q), c
        sel[d] = best
    from collections import Counter
    log(f"  E2[{q_mode}] 变体分布: {Counter(sel[31:365]).most_common(6)}")

    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = 6000.0
    t0 = time.time()
    for d in range(DAYS):
        p, q = sel[d]
        lf, pf = preds[p]
        net_q = Q[(p, q)][d]
        if np.isnan(net_q).any():
            net_q = lf[d] - pf[d]
        soc_start[d] = soc
        sol = solve_plan(price[d], net_q + pf[d], pf[d], soc0=soc, cyclic=True,
                         validate=False)
        g = sol["g"]
        g_plan[d] = g
        cost_plan[d] = float(price[d] @ g)
        net = g + PV[d] * DT - LOAD[d] * DT
        for k in range(N):
            nk = net[k]
            if nk >= 0:
                ch = min(nk, 5000 * DT, (10800 - soc) / 0.9)
                dis, em, sp = 0.0, 0.0, nk - ch
            else:
                deficit = -nk
                if exec_mode == "rolling" and soc > 1200 + 1e-6:
                    load_k = np.concatenate([[LOAD[d, k]], lf[d, k + 1:]])
                    pv_k = np.concatenate([[PV[d, k]], pf[d, k + 1:]])
                    ch, dis, em, sp = exec_lp(price[d][k:], load_k, pv_k, g[k:],
                                               soc, soc_start[d])
                else:
                    dis = min(deficit, 5000 * DT, (soc - 1200) * 0.9)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += 0.9 * ch - dis / 0.9
        cost_emg[d] = float(price[d] @ EM[d]) * 5.0
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg,
               cost_total=cost_plan + cost_emg)
    log(f"  E2[{q_mode}]: 回测期总费用 {out['cost_total'][REP].sum():,.0f} 元 "
        f"(计划 {out['cost_plan'][REP].sum():,.0f}, 紧急 {out['cost_emg'][REP].sum():,.0f}), "
        f"{time.time() - t0:.1f}s")
    return out


def _run_saa(job):
    tag, rule, S = job
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    return tag, simulate_saa_group(lf, pf, PRICE_A1[None, :], rule=rule, S=S)


def _run_pool(job):
    tag, qm = job
    return tag, simulate_quantile_pool(PRICE_A1[None, :], q_mode=qm)


def main():
    from concurrent.futures import ProcessPoolExecutor
    saa_jobs = [("E7_weighted", "weighted", 14), ("E7_mix10sw2", "mix10sw2", 12)]
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_run_saa, j) for j in saa_jobs]
        results = {t: r for t, r in [f.result() for f in futs]}
    print("\n===== E7 加权场景池（回测期 2025.2.1-12.31）=====")
    base = 13938354.0
    for tag in sorted(results):
        tot = float(results[tag]["cost_total"][REP].sum())
        print(f"{tag:<12}{tot:>15,.0f} 元   vs 当前主策略 {tot - base:+,.0f} ({(tot - base) / base * 100:+.2f}%)")
    import json
    (RESULTS / "e7_results.json").write_text(json.dumps(
        {t: float(r["cost_total"][REP].sum()) for t, r in results.items()},
        indent=2), encoding="utf-8")
    best_tag = min(results, key=lambda t: float(results[t]["cost_total"][REP].sum()))
    tot_best = float(results[best_tag]["cost_total"][REP].sum())
    if tot_best < base:
        np.savez(RESULTS / "e_best_sim.npz", **{k: results[best_tag][k] for k in
                 ["g_plan", "c", "d", "spill", "emg", "soc_start", "cost_plan", "cost_emg"]})
        print(f"\n最优: {best_tag} ({tot_best:,.0f}) → 已存 results/e_best_sim.npz")
    else:
        print(f"\n最优: {best_tag} ({tot_best:,.0f}) 仍差于当前主策略，不采纳")


if __name__ == "__main__":
    main()
