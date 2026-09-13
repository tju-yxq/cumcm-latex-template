# -*- coding: utf-8 -*-
"""问题2：随机负载/光伏下的日前计划 + 实时执行仿真。

流程（严格因果）：
  每天 0:00 —— 用历史数据预测当日负载/光伏（q2_forecast 的 hybrid 预测器）
            —— 净负荷预测误差的近28天逐时段 q 分位数缓冲（报童模型：5倍惩罚 ⇒ F(q)=0.8）
            —— 解日前 LP（负载预测+缓冲）得计划购电量 g（按计划量结算 take-or-pay）
  白天每 10 分钟 —— 实际净负荷 vs g：盈余充电(越限弃)、缺口放电(越限紧急购电 5×)
  储能 SOC 跨日连续传递；1月为过渡期，2.1-12.31（334天）为正式回测期。

执行模式：greedy（缺口即放电）；rolling（缺口时对剩余时段重解 LP，权衡当前 5p_k
放电与储存电量规避未来更高价 5p 的紧急购电；盈余时段贪心充电恒为最优，无需求解）。
"""
import sys, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from common import (load_all, solve_plan, key_numbers, log, RESULTS,
                    N, DT, E_MAX, SOC_MIN, SOC_MAX, ETA, SOC_INIT, DAYS)

A1, LOAD, PV, PRICE4, FC3 = load_all()
PRICE_A1 = A1["price"]                      # 固定日电价（文件列序，每天相同）
JAN = slice(0, 31)
REP = slice(31, 365)                        # 2025.2.1 - 12.31


def buffer_curve(net_err_hist, q):
    """net_err_hist: (n,144) 净负荷预测误差（预测-实际, kW）。
    紧急购电由「实际净负荷 > 计划供给」触发，故缓冲取 (实际-预测) 的 q 分位数。"""
    if q is None or len(net_err_hist) == 0:
        return np.zeros(N)
    return np.maximum(np.quantile(-net_err_hist[-28:], q, axis=0), 0.0)


def exec_lp(price_k, load_kw_k, pv_kw_k, g_lock_k, soc, soc_target):
    """滚动执行：剩余时段 LP（g 锁定；时段0用实际值，未来用当日点预测）。
    返回时段0的 (charge, discharge, emergency, spill)。"""
    n = len(g_lock_k)
    nv = 6 * n
    G, C, D_, SP, EM, S = [slice(i * n, (i + 1) * n) for i in range(6)]
    obj = np.zeros(nv)
    obj[EM] = 5.0 * price_k
    obj[C] = 1e-7; obj[D_] = 1e-7                 # 微罚吞吐：消除同时充放的退化解
    load_e = np.asarray(load_kw_k) * DT
    pv_e = np.asarray(pv_kw_k) * DT
    A = lil_matrix((2 * n + 1, nv)); b = np.zeros(2 * n + 1)
    for t in range(n):
        A[t, S.start + t] = 1.0
        if t > 0:
            A[t, S.start + t - 1] = -1.0
        A[t, C.start + t] = -ETA
        A[t, D_.start + t] = 1.0 / ETA
    b[0] = soc
    for t in range(n):
        r = n + t
        A[r, G.start + t] = 1.0; A[r, D_.start + t] = 1.0; A[r, EM.start + t] = 1.0
        A[r, C.start + t] = -1.0; A[r, SP.start + t] = -1.0
        b[r] = load_e[t] - pv_e[t]
    A[2 * n, S.start + n - 1] = 1.0; b[2 * n] = soc_target   # 终端：当日循环水平
    bounds = [(0, None)] * nv
    for t in range(n):
        bounds[C.start + t] = (0, E_MAX); bounds[D_.start + t] = (0, E_MAX)
        bounds[S.start + t] = (SOC_MIN, SOC_MAX)
        bounds[G.start + t] = (g_lock_k[t], g_lock_k[t])
    res = linprog(obj, A_eq=A.tocsr(), b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:                                       # 终端不可行 → 放松
        res = linprog(obj, A_eq=A.tocsr()[: 2 * n], b_eq=b[: 2 * n],
                      bounds=bounds, method="highs")
        assert res.status == 0, "执行 LP 不可行"
    x = res.x
    return x[C][0], x[D_][0], x[EM][0], x[SP][0]


def simulate(load_fc, pv_fc, price_days, buffer_q=0.8, exec_mode="greedy"):
    """全年 365 天仿真。price_days: (DAYS,144) 或 (1,144)。"""
    price = np.broadcast_to(np.asarray(price_days, float), (DAYS, N))
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS)
    cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    net_err_hist = []
    t0 = time.time()
    for d in range(DAYS):
        lf, pf = load_fc[d], pv_fc[d]
        buf = buffer_curve(np.array(net_err_hist) if net_err_hist else np.zeros((0, N)), buffer_q)
        soc_start[d] = soc
        sol = solve_plan(price[d], lf + buf, pf, soc0=soc, cyclic=True, validate=(d < 3))
        assert sol["status"] == 0, f"day {d} 计划 LP 失败"
        g = sol["g"]
        g_plan[d] = g
        cost_plan[d] = float(price[d] @ g)

        net = g + PV[d] * DT - LOAD[d] * DT                # kWh，计划日槽坐标
        for k in range(N):
            nk = net[k]
            if nk >= 0:                                     # 盈余：贪心充电恒最优
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
        net_err_hist.append((lf - pf) - (LOAD[d] - PV[d]))

    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg, cost_total=cost_plan + cost_emg,
               spill_total=float(SP.sum()), emg_total=float(EM.sum()))
    log(f"  仿真({exec_mode}, q={buffer_q}): 回测期总费用 {out['cost_total'][REP].sum():,.0f} 元, "
        f"回测期紧急 {out['emg'][REP].sum():,.0f} kWh, {time.time() - t0:.1f}s")
    return out


def clock_day_view(arr_pd):
    """计划日坐标 (DAYS,144) → 钟表日坐标 (DAYS+1,144)。
    全局槽 m=d*144+k+1 → clock day=m//144, j=m%144（末槽越界到第366天，截断）。"""
    out = np.zeros((DAYS + 1, N))
    for d in range(DAYS):
        for k in range(N):
            m = d * N + k + 1
            out[m // N, m % N] = arr_pd[d, k]
    return out[:DAYS]


def emergency_intervals(emg_clock, D):
    """钟表日 D 紧急购电连续区间合并 → [(label, kWh)]"""
    slots = np.where(emg_clock[D] > 1e-6)[0]
    if len(slots) == 0:
        return []
    ivs, st, prev = [], slots[0], slots[0]
    for j in slots[1:]:
        if j == prev + 1:
            prev = j
        else:
            ivs.append((st, prev)); st = prev = j
    ivs.append((st, prev))
    fmt = lambda m: f"{m // 60}:{m % 60:02d}" if m < 1440 else "24:00"
    return [(f"{fmt(a * 10)}-{fmt((b + 1) * 10)}", float(emg_clock[D, a:b + 1].sum()))
            for a, b in ivs]


def main():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    load_fc, pv_fc = fc["load_fc"], fc["pv_fc"]          # hybrid 预测器

    # ---- 1月（过渡期）选缓冲分位数：报童理论值 0.8 附近网格 ----
    log("【1月调参】缓冲分位数网格（hybrid 预测器 + 贪心执行，按 1 月总费用选择）")
    jan_scores = {}
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        sim = simulate(load_fc, pv_fc, PRICE_A1[None, :], buffer_q=q, exec_mode="greedy")
        jan_scores[q] = float(sim["cost_total"][JAN].sum())
        print(f"  q={q}: 1月总费用 {jan_scores[q]:,.0f} 元")
    q_star = min(jan_scores, key=jan_scores.get)
    log(f"  → 选定 q* = {q_star}（理论值 0.8）")

    # ---- 正式回测（2.1-12.31）四策略 ----
    log("【正式回测 2.1-12.31】")
    base_cost = np.array([float(PRICE_A1 @ np.maximum(LOAD[d] * DT - PV[d] * DT, 0))
                          for d in range(DAYS)])
    sim_naive = simulate(load_fc, pv_fc, PRICE_A1[None, :], buffer_q=None, exec_mode="greedy")
    sim_main = simulate(load_fc, pv_fc, PRICE_A1[None, :], buffer_q=q_star, exec_mode="greedy")
    sim_roll = simulate(load_fc, pv_fc, PRICE_A1[None, :], buffer_q=q_star, exec_mode="rolling")
    sim_oracle = simulate(LOAD, PV, PRICE_A1[None, :], buffer_q=None, exec_mode="greedy")

    R = REP
    rows = [
        ("无储能基准", float(base_cost[R].sum()), 0.0),
        ("朴素预测（无缓冲）+贪心", float(sim_naive["cost_total"][R].sum()), float(sim_naive["cost_emg"][R].sum())),
        (f"分位数缓冲(q={q_star})+贪心", float(sim_main["cost_total"][R].sum()), float(sim_main["cost_emg"][R].sum())),
        (f"分位数缓冲(q={q_star})+滚动执行", float(sim_roll["cost_total"][R].sum()), float(sim_roll["cost_emg"][R].sum())),
        ("理想先知（完全预测）", float(sim_oracle["cost_total"][R].sum()), float(sim_oracle["cost_emg"][R].sum())),
    ]
    print("\n策略 | 回测期总费用(元) | 其中紧急购电费(元) | 紧急占比")
    for name, tot, em in rows:
        print(f"{name:<26}{tot:>15,.0f}{em:>17,.0f}   {em / max(tot, 1) * 100:>5.1f}%")

    best = sim_roll if sim_roll["cost_total"][R].sum() < sim_main["cost_total"][R].sum() else sim_main
    best_mode = "rolling" if best is sim_roll else "greedy"
    log(f"主策略执行模式: {best_mode}")

    # ---- 缓冲分位数全年敏感性 ----
    sens_q = []
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        if q == q_star:
            c = float(best["cost_total"][R].sum())
        elif q == 0.5:
            c = float(sim_naive["cost_total"][R].sum())
        else:
            c = float(simulate(load_fc, pv_fc, PRICE_A1[None, :], buffer_q=q,
                               exec_mode=best_mode)["cost_total"][R].sum())
        sens_q.append((q, c))
        print(f"  敏感性 q={q}: 回测期总费用 {c:,.0f} 元")

    np.savez(RESULTS / "q2_main_sim.npz",
             **{k: best[k] for k in ["g_plan", "c", "d", "spill", "emg",
                                     "soc_start", "cost_plan", "cost_emg"]})
    key_numbers(
        q2_buffer_q=q_star, q2_exec_mode=best_mode,
        q2_cost_baseline=round(float(base_cost[R].sum()), 0),
        q2_cost_naive=round(float(sim_naive["cost_total"][R].sum()), 0),
        q2_cost_main=round(float(best["cost_total"][R].sum()), 0),
        q2_cost_main_plan=round(float(best["cost_plan"][R].sum()), 0),
        q2_cost_main_emg=round(float(best["cost_emg"][R].sum()), 0),
        q2_cost_oracle=round(float(sim_oracle["cost_total"][R].sum()), 0),
        q2_cost_greedy=round(float(sim_main["cost_total"][R].sum()), 0),
        q2_cost_rolling=round(float(sim_roll["cost_total"][R].sum()), 0),
        q2_emg_total_kwh=round(float(best["emg"][REP].sum()), 0),
        q2_spill_total_kwh=round(float(best["spill"][REP].sum()), 0),
        q2_sens_q={str(q): round(c, 0) for q, c in sens_q},
        q2_jan_scores={str(q): round(v, 0) for q, v in jan_scores.items()},
    )
    log("主策略仿真与关键数值已保存")
    return best, q_star, best_mode


if __name__ == "__main__":
    main()
