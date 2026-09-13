# -*- coding: utf-8 -*-
"""问题3：多时点预报下的滚动调整购电模型。

信息结构：0:00 有附件3 的 0:00 光伏预报（24 整点线性插值）+ 自建负载预测；
6/12/18 时各有新预报，可对 T+1 时起的剩余时段做调整。
偏差结算（主口径 A）：最终量 a 相对 0:00 计划 g，
  结算 = p·a + 0.5p·(a-g)^+ + 0.5p·(g-a)^+  （少买部分按 50% 结算、多买超出部分 1.5 倍）
备选口径 B：结算 = p·a + 0.5p·(a-g)^+ + 1.5p·(g-a)^+  （少买不退款，敏感性分析）。
调整 LP 中以 u=a-g 上偏、v=g-a 下偏线性化：A 口标 0.5p·(u+v)，B 口标 0.5p·u+1.5p·v。
缓冲：各发布时刻分别用其净负荷预测误差的近 28 天 q 分位数（报童模型，q 沿用问题二协议 0.7）。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from common import (load_all, solve_plan, key_numbers, log, RESULTS,
                    N, DT, E_MAX, SOC_MIN, SOC_MAX, ETA, SOC_INIT, DAYS)
from common import pv_profile_day
from q2_simulate import exec_lp, PRICE_A1, REP, clock_day_view

A1, LOAD, PV, PRICE4, FC3 = load_all()
BUF_Q = 0.7
ISSUES = {6: 1, 12: 2, 18: 3}          # 发布时刻 -> fc3 列
K0 = {6: 41, 12: 77, 18: 113}          # 各发布时刻首个可调槽（T+1h 的标签槽）


def build_profiles():
    load_fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")["load_fc"]
    mean7 = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")["mean7_pv"]
    off0 = np.array([pv_profile_day(FC3[d, 0, :], 0) for d in range(DAYS)])
    offT = {T: np.array([pv_profile_day(FC3[d, i, :], T) for d in range(DAYS)])
            for T, i in ISSUES.items()}

    def blend(off, k0, k1):
        """因果逆方差组合：官方预报 × 历史均值，逐槽权重由近28天误差方差估计。
        官方预报噪声大（0:00发布 MAE≈199）而历史均值偏平滑（对午后差），
        组合后 MAE 全面低于单一来源（128.7/132.2/58.1 vs 各自最优 154.5/161.4/65.6）。"""
        out = off.copy()
        for d in range(1, DAYS):
            h = slice(max(0, d - 28), d)
            v1 = ((off[h, k0:k1] - PV[h, k0:k1]) ** 2).mean(0) + 1e-9
            v2 = ((mean7[h, k0:k1] - PV[h, k0:k1]) ** 2).mean(0) + 1e-9
            w = v2 / (v1 + v2)                     # 官方预报权重
            out[d, k0:k1] = w * off[d, k0:k1] + (1 - w) * mean7[d, k0:k1]
        return out

    pv0 = blend(off0, 0, 143)
    pvT = {T: blend(offT[T], K0[T], 144) for T in ISSUES}
    return load_fc, pv0, pvT, off0, mean7


def build_buffers(load_fc, pv0, pvT):
    """各发布时刻的净负荷误差分位数缓冲（严格因果：只用当天之前）。
    注意方向：紧急购电由「实际净负荷 > 预测」触发，缓冲取 (实际-预测) 的 q 分位数。"""
    act_net = LOAD - PV
    buf0 = np.zeros((DAYS, N))
    for d in range(DAYS):
        if d == 0:
            continue
        err = act_net[:d] - (load_fc[:d] - pv0[:d])     # 实际 - 预测
        buf0[d] = np.maximum(np.quantile(err[max(0, d - 28):], BUF_Q, axis=0), 0.0)
    bufT = {T: np.zeros((DAYS, N)) for T in ISSUES}
    for T in ISSUES:
        k0 = K0[T]
        for d in range(DAYS):
            if d == 0:
                continue
            hist = []
            for j in range(max(0, d - 28), d):
                e = act_net[j, k0:] - (load_fc[j, k0:] - pvT[T][j, k0:])
                hist.append(e)
            if hist:
                bufT[T][d, k0:] = np.maximum(np.quantile(np.array(hist), BUF_Q, axis=0), 0.0)
    return buf0, bufT


def build_errT(load_fc, pvT):
    """各发布时刻新预报的净负荷预测误差（预测-实际，kW），供场景化调整使用。
    errT[T][d, k0:] = (load_fc[d] - pvT[T][d]) - act_net[d]，覆盖槽之外为 NaN。"""
    act_net = LOAD - PV
    errT = {}
    for T in ISSUES:
        k0 = K0[T]
        e = np.full((DAYS, N), np.nan)
        e[:, k0:] = (load_fc[:, k0:] - pvT[T][:, k0:]) - act_net[:, k0:]
        errT[T] = e
    return errT


def adjust_lp_saa(price_k, net_fc_k, errs, g_plan_k, soc, soc_target, coeff_v=0.5):
    """T 时刻场景化两阶段调整 LP（与 0:00 的 SAA 计划层风险视角一致）。

    第一阶段：调整购电量 a = g + u - v（偏差结算 p·a + 0.5p·u + coeff_v·p·v）；
    第二阶段：近 S 天新预报误差场景下的充/放/弃/紧急（期望罚金 (1/S)Σ5p·e^s）。
    net_fc_k: (n,) 新预报净负荷(kW)；errs: (S,n) 历史误差(预测-实际,kW)。返回 a。"""
    S = len(errs)
    n = len(g_plan_k)
    if S == 0:
        return g_plan_k.copy()                     # 无历史（过渡期）→ 不调整
    nE = (np.asarray(net_fc_k)[None, :] - errs) * DT       # (S,n) 场景净负荷 kWh
    NV = 3 * n + 5 * n * S
    A_, U, V = 0, n, 2 * n
    C = lambda s: 3 * n + 5 * n * s
    D_ = lambda s: 3 * n + 5 * n * s + n
    SP = lambda s: 3 * n + 5 * n * s + 2 * n
    EM = lambda s: 3 * n + 5 * n * s + 3 * n
    S_ = lambda s: 3 * n + 5 * n * s + 4 * n

    obj = np.zeros(NV)
    obj[A_:A_ + n] = price_k
    obj[U:U + n] = 0.5 * price_k
    obj[V:V + n] = coeff_v * price_k
    for s in range(S):
        obj[EM(s):EM(s) + n] = 5.0 * price_k / S
        obj[C(s):C(s) + n] = 1e-7
        obj[D_(s):D_(s) + n] = 1e-7

    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(n):                                    # a = g + u - v
        rows += [r, r, r]; cols += [A_ + t, U + t, V + t]; vals += [1.0, -1.0, 1.0]
        beq.append(g_plan_k[t]); r += 1
    for s in range(S):
        for t in range(n):                                # SOC
            rows.append(r); cols.append(S_(s) + t); vals.append(1.0)
            if t > 0:
                rows.append(r); cols.append(S_(s) + t - 1); vals.append(-1.0)
            else:
                beq.append(soc)
            rows += [r, r]; cols += [C(s) + t, D_(s) + t]; vals += [-ETA, 1.0 / ETA]
            if t > 0:
                beq.append(0.0)
            r += 1
        rows.append(r); cols.append(S_(s) + n - 1); vals.append(1.0)
        beq.append(soc_target); r += 1                    # 终端日循环
        for t in range(n):                                # 平衡
            rows += [r, r, r, r, r]
            cols += [A_ + t, D_(s) + t, EM(s) + t, C(s) + t, SP(s) + t]
            vals += [1.0, 1.0, 1.0, -1.0, -1.0]
            beq.append(nE[s, t]); r += 1
    from scipy.sparse import coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(r, NV)).tocsr()
    b = np.array(beq)
    bounds = [(0, None)] * NV
    for s in range(S):
        for t in range(n):
            bounds[C(s) + t] = (0, E_MAX)
            bounds[D_(s) + t] = (0, E_MAX)
            bounds[S_(s) + t] = (SOC_MIN, SOC_MAX)
    res = linprog(obj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:                                   # 终端不可行 → 去掉各场景终端行
        drop = set()
        idx = n
        for s in range(S):
            drop.add(idx + n)
            idx += 2 * n + 1
        keep = [i for i in range(r) if i not in drop]
        A2 = A[keep, :]
        b2 = b[keep]
        res = linprog(obj, A_eq=A2, b_eq=b2, bounds=bounds, method="highs")
        assert res.status == 0, "场景化调整 LP 不可行"
    return res.x[A_:A_ + n]


def simulate_q3(load_fc, pv0, pvT, buf0, bufT, adjust_times, exec_mode="rolling",
                coeff_v=0.5, price_days=None, plan_mode="buffer", errT=None,
                scenario_rule="recent"):
    """问题3 全年仿真。plan_mode: "buffer"/"saa" 为 0:00 计划方式；
    scenario_rule: SAA 与调整层的场景日选择（"recent"=近10天混合（默认，实测最优）；
    "group"=按负荷日型分组（周五/周六为低需求组——实验证伪：混合场景对低需求日的
    隐性过度防御反而更优，保留参数仅作对照））。调整层为场景化两阶段 LP。"""
    price = np.broadcast_to(PRICE_A1 if price_days is None else np.asarray(price_days, float),
                            (DAYS, N))
    G = np.zeros((DAYS, N)); Afin = np.zeros((DAYS, N))
    C = np.zeros((DAYS, N)); D_ = np.zeros((DAYS, N))
    SP = np.zeros((DAYS, N)); EM = np.zeros((DAYS, N))
    soc_start = np.zeros(DAYS)
    cost_settle = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    err_hist_saa = []                          # SAA 计划层的净负荷误差流（预测-实际）
    t0 = time.time()
    for d in range(DAYS):
        lf, p0 = load_fc[d], pv0[d]
        price_d = price[d]
        soc_start[d] = soc
        if plan_mode == "saa":
            from q2_saa import solve_saa_plan
            from q2_variants import scenario_errors_days
            err_all = np.array(err_hist_saa) if err_hist_saa else np.zeros((0, N))
            errs_sel = scenario_errors_days(err_all, d, scenario_rule, 10)
            if len(errs_sel):
                g = solve_saa_plan(price_d, lf, p0, errs_sel, soc, S=len(errs_sel)).copy()
            else:
                sol = solve_plan(price_d, lf, p0, soc0=soc, cyclic=True, validate=False)
                g = sol["g"].copy()
        else:
            sol = solve_plan(price_d, lf + buf0[d], p0, soc0=soc, cyclic=True, validate=(d < 3))
            assert sol["status"] == 0
            g = sol["g"].copy()
        G[d] = g
        cur_g = g.copy()
        cur_pv = p0.copy()                               # 当前光伏预测（最新发布）
        cur_buf = buf0[d].copy()
        for k in range(N):
            for T in adjust_times:
                if k == K0[T]:
                    newpv = pvT[T][d].copy()
                    mask = ~np.isnan(newpv)
                    cur_pv[mask] = newpv[mask]
                    k0 = K0[T]
                    if errT is not None and d >= 11:
                        net_new = (lf - cur_pv)[k0:]
                        if scenario_rule == "group":
                            from q2_variants import GRP
                            days = [j for j in range(max(1, d - 21), d)
                                    if GRP[j] == GRP[d]][-10:]
                        else:
                            days = list(range(max(1, d - 10), d))
                        errs = errT[T][days, k0:]
                        a = adjust_lp_saa(price_d[k0:], net_new, errs, g[k0:],
                                          soc, soc_start[d], coeff_v)
                    else:                      # 过渡期无误差历史 → 不调整
                        a = g[k0:]
                    cur_g[k0:] = np.maximum(a, 0.0)
            # ---- 执行槽 k ----
            net = cur_g[k] + PV[d, k] * DT - LOAD[d, k] * DT
            if net >= 0:
                ch = min(net, E_MAX, (SOC_MAX - soc) / ETA)
                dis, em, sp = 0.0, 0.0, net - ch
            else:
                deficit = -net
                if exec_mode == "rolling" and soc > SOC_MIN + 1e-6:
                    load_k = np.concatenate([[LOAD[d, k]], lf[k + 1:]])
                    pv_k = np.concatenate([[PV[d, k]], cur_pv[k + 1:]])
                    ch, dis, em, sp = exec_lp(price_d[k:], load_k, pv_k, cur_g[k:],
                                               soc, soc_start[d])
                else:
                    dis = min(deficit, E_MAX, (soc - SOC_MIN) * ETA)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += ETA * ch - dis / ETA
        Afin[d] = cur_g
        # 结算（口径 A/B 由 coeff_v 决定记录）
        dev = cur_g - g
        if coeff_v == 0.5:
            cost_settle[d] = float(price_d @ cur_g + 0.5 * price_d * np.abs(dev) @ np.ones(N))
        else:
            cost_settle[d] = float(price_d @ cur_g) + float(
                (0.5 * price_d * np.maximum(dev, 0)).sum() +
                (1.5 * price_d * np.maximum(-dev, 0)).sum())
        cost_emg[d] = float(price_d @ EM[d]) * 5.0
        err_hist_saa.append((lf - p0) - (LOAD[d] - PV[d]))

    out = dict(g_plan=G, a_final=Afin, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_settle=cost_settle, cost_emg=cost_emg,
               cost_total=cost_settle + cost_emg,
               settle_reading="A(退款)" if coeff_v == 0.5 else "B(不退款)")
    log(f"  Q3仿真(调整={adjust_times or '无'}, {exec_mode}): 回测期总费用 "
        f"{out['cost_total'][REP].sum():,.0f} 元 (结算 {out['cost_settle'][REP].sum():,.0f}, "
        f"紧急 {out['cost_emg'][REP].sum():,.0f}), {time.time() - t0:.1f}s")
    return out


def _run_strategy(job):
    """并行 worker：跑单个策略配置。"""
    name, at, cv, load_fc, pv0, pvT, buf0, bufT, errT = job
    return name, simulate_q3(load_fc, pv0, pvT, buf0, bufT, at, exec_mode="rolling",
                             plan_mode="saa", errT=errT, coeff_v=cv)


def main():
    load_fc, pv0, pvT, off0, mean7 = build_profiles()
    # 光伏预测精度：官方预报 vs 历史均值 vs 逆方差组合（正式期）
    rep = slice(31, 365)
    mae_off = np.abs((off0[rep] - PV[rep])).mean()
    mae_m7 = np.abs((mean7[rep] - PV[rep])).mean()
    mae_bl = np.abs((pv0[rep] - PV[rep])).mean()
    log(f"光伏预测精度(0:00信息): 官方预报 MAE={mae_off:.1f} vs 历史均值 {mae_m7:.1f} "
        f"vs 逆方差组合 {mae_bl:.1f} kW")
    buf0, bufT = build_buffers(load_fc, pv0, pvT)
    errT = build_errT(load_fc, pvT)

    # ---- 是否需要其他时刻预报：策略对比（SAA 计划层 + 场景化调整层，多进程并行） ----
    strategies = {
        "A_不调整": [],
        "B_仅6时": [6],
        "C_仅12时": [12],
        "D_仅18时": [18],
        "E_全调整(主)": [6, 12, 18],
    }

    jobs = {name: (name, at, 0.5, load_fc, pv0, pvT, buf0, bufT, errT)
            for name, at in strategies.items()}
    jobs["口径B"] = ("口径B", [6, 12, 18], 1.5, load_fc, pv0, pvT, buf0, bufT, errT)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(6, len(jobs))) as ex:
        results = dict(ex.map(_run_strategy, jobs.values()))
    sims = {k: v for k, v in results.items() if k != "口径B"}

    print("\n策略 | 回测期总费用(元) | 偏差结算费 | 紧急购电费 | 较不调整节省")
    base = sims["A_不调整"]["cost_total"][REP].sum()
    for name, s in sims.items():
        tot = s["cost_total"][REP].sum()
        print(f"{name:<14}{tot:>15,.0f}{s['cost_settle'][REP].sum():>13,.0f}"
              f"{s['cost_emg'][REP].sum():>13,.0f}{base - tot:>13,.0f}")

    main_sim = sims["E_全调整(主)"]
    sim_B = results["口径B"]                      # 结算口径 B 敏感性（并行已跑完）

    np.savez(RESULTS / "q3_main_sim.npz",
             **{k: main_sim[k] for k in ["g_plan", "a_final", "c", "d", "spill",
                                         "emg", "soc_start", "cost_settle", "cost_emg"]})
    key_numbers(
        q3_pv_mae_official=round(float(mae_off), 1), q3_pv_mae_self=round(float(mae_m7), 1),
        q3_pv_mae_blend=round(float(mae_bl), 1),
        q3_cost_noAdj=round(float(sims["A_不调整"]["cost_total"][REP].sum()), 0),
        q3_cost_adj6=round(float(sims["B_仅6时"]["cost_total"][REP].sum()), 0),
        q3_cost_adj12=round(float(sims["C_仅12时"]["cost_total"][REP].sum()), 0),
        q3_cost_adj18=round(float(sims["D_仅18时"]["cost_total"][REP].sum()), 0),
        q3_cost_main=round(float(main_sim["cost_total"][REP].sum()), 0),
        q3_main_settle=round(float(main_sim["cost_settle"][REP].sum()), 0),
        q3_main_emg=round(float(main_sim["cost_emg"][REP].sum()), 0),
        q3_cost_readingB=round(float(sim_B["cost_total"][REP].sum()), 0),
        q3_readingB_settle=round(float(sim_B["cost_settle"][REP].sum()), 0),
        q3_readingB_emg=round(float(sim_B["cost_emg"][REP].sum()), 0),
        q3_emg_total_kwh=round(float(main_sim["emg"][REP].sum()), 0),
        q3_adj_total_kwh=round(float(np.abs(main_sim["a_final"][REP] -
                                            main_sim["g_plan"][REP]).sum()), 0),
    )
    log("Q3 主策略与关键数值已保存")
    return main_sim


if __name__ == "__main__":
    main()
