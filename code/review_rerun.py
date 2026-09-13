# -*- coding: utf-8 -*-
"""审稿重跑实验 #6（紧急充电禁止）+ #1（P3时点提前）+ #3（0:00/0:10偏移量化）。"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import load_all, solve_plan, log, RESULTS, N, DT, E_MAX, SOC_MIN, SOC_MAX, ETA, SOC_INIT, DAYS
from q2_saa import solve_saa_plan
from q2_simulate import exec_lp, PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()
fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf, pf = fc["load_fc"], fc["pv_fc"]
err_all = (lf - pf) - (LOAD - PV)


# ================================================================
# #6: 紧急购电禁止充电——执行层禁止 e_t > 0 时 c_t > 0
# ================================================================
def simulate_no_emerg_charge(S=10):
    """SAA 计划 + 执行层禁止紧急购电用于充电（缺口时放电，不够就紧急但不充电）。"""
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    t0 = time.time()
    for d in range(DAYS):
        soc_start[d] = soc
        errs = err_all[max(0, d - S):d] if d else np.zeros((0, N))
        if len(errs):
            g = solve_saa_plan(PRICE_A1, lf[d], pf[d], errs, soc, S=len(errs))
        else:
            g = solve_plan(PRICE_A1, lf[d], pf[d], soc0=soc, cyclic=True, validate=False)["g"]
        g_plan[d] = g
        cost_plan[d] = float(PRICE_A1 @ g)
        net = g + PV[d] * DT - LOAD[d] * DT
        for k in range(N):
            nk = net[k]
            if nk >= 0:
                # 盈余：充电（禁止紧急）
                ch = min(nk, E_MAX, (SOC_MAX - soc) / ETA)
                dis, em, sp = 0.0, 0.0, nk - ch
            else:
                # 缺口：放电 → 不够就紧急（禁止同时充电）
                deficit = -nk
                dis = min(deficit, E_MAX, (soc - SOC_MIN) * ETA)
                em = deficit - dis
                ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += ETA * ch - dis / ETA
        cost_emg[d] = float(PRICE_A1 @ EM[d]) * 5.0
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg, cost_total=cost_plan + cost_emg)
    tot = float(out["cost_total"][REP].sum())
    log(f"  #6 禁止紧急充电: {tot:,.0f} 元 (计划 {float(out['cost_plan'][REP].sum()):,.0f}, "
        f"紧急 {float(out['cost_emg'][REP].sum()):,.0f}), {time.time()-t0:.0f}s")
    return out


# ================================================================
# #1: P3 调整时点提前到 T 时刻（用 T 时 SOC + 预测 T→T+1 SOC）
# ================================================================
def simulate_p3_early_adjust(adjust_times=(6, 12, 18)):
    """P3：调整在 T 时触发（而非 T+1），用 T 时实际 SOC + 预测 T→T+1 SOC 演化。"""
    from q3_simulate import build_profiles, build_errT, K0, adjust_lp_saa
    lf3, pv0, pvT, _, _ = build_profiles()
    buf0_unused, bufT_unused = None, None
    errT = build_errT(lf3, pvT)
    price = PRICE_A1
    G = np.zeros((DAYS, N)); Afin = np.zeros((DAYS, N))
    C = np.zeros((DAYS, N)); D_ = np.zeros((DAYS, N))
    SP = np.zeros((DAYS, N)); EM = np.zeros((DAYS, N))
    soc_start = np.zeros(DAYS); cost_settle = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = SOC_INIT
    err_hist_saa = []
    t0 = time.time()
    for d in range(DAYS):
        lf, p0 = lf3[d], pv0[d]
        soc_start[d] = soc
        # 0:00 计划（SAA）
        err_all_p3 = np.array(err_hist_saa) if err_hist_saa else np.zeros((0, N))
        if len(err_all_p3):
            errs_sel = err_all_p3[max(0, d-10):d]
            g = solve_saa_plan(price, lf, p0, errs_sel, soc, S=len(errs_sel)).copy()
        else:
            g = solve_plan(price, lf, p0, soc0=soc, cyclic=True, validate=False)["g"].copy()
        G[d] = g
        cur_g = g.copy()
        cur_pv = p0.copy()

        for k in range(N):
            # 在 T 时刻（而非 T+1）触发调整
            for T in adjust_times:
                trigger_slot = K0[T] - 6  # T 时刻对应的槽号
                if k == trigger_slot and d >= 11:
                    # 用当前 SOC（T 时刻）
                    # 预测 T→T+1 的 SOC 演化（用计划值+光伏预测）
                    soc_at_t1 = soc
                    for j in range(trigger_slot, K0[T]):
                        net_j = cur_g[j] + cur_pv[j] * DT - lf[j] * DT  # 用预测
                        if net_j >= 0:
                            ch_j = min(net_j, E_MAX, (SOC_MAX - soc_at_t1) / ETA)
                            soc_at_t1 += ETA * ch_j
                        else:
                            dis_j = min(-net_j, E_MAX, (soc_at_t1 - SOC_MIN) * ETA)
                            soc_at_t1 -= dis_j / ETA
                    # 用预测的 T+1 SOC 解调整 LP
                    k0 = K0[T]
                    # 更新 cur_pv（与原 q3_simulate 相同逻辑）
                    newpv = pvT[T][d].copy()
                    mask = ~np.isnan(newpv)
                    cur_pv[mask] = newpv[mask]
                    net_new = (lf - cur_pv)[k0:]
                    days = list(range(max(1, d - 10), d))
                    errs = errT[T][days, k0:]
                    a = adjust_lp_saa(price[k0:], net_new, errs, g[k0:],
                                      soc_at_t1, soc_start[d], 0.5)
                    cur_g[k0:] = np.maximum(a, 0.0)

            # 执行槽 k
            net = cur_g[k] + PV[d, k] * DT - LOAD[d, k] * DT
            if net >= 0:
                ch = min(net, E_MAX, (SOC_MAX - soc) / ETA)
                dis, em, sp = 0.0, 0.0, net - ch
            else:
                deficit = -net
                if soc > SOC_MIN + 1e-6:
                    load_k = np.concatenate([[LOAD[d, k]], lf[k + 1:]])
                    pv_k = np.concatenate([[PV[d, k]], cur_pv[k + 1:]])
                    ch, dis, em, sp = exec_lp(price[k:], load_k, pv_k, cur_g[k:],
                                               soc, soc_start[d])
                else:
                    dis = min(deficit, E_MAX, (soc - SOC_MIN) * ETA)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += ETA * ch - dis / ETA

        Afin[d] = cur_g
        dev = cur_g - g
        cost_settle[d] = float(price @ cur_g + 0.5 * price * np.abs(dev) @ np.ones(N))
        cost_emg[d] = float(price @ EM[d]) * 5.0
        err_hist_saa.append((lf - p0) - (LOAD[d] - PV[d]))

    out = dict(g_plan=G, a_final=Afin, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_settle=cost_settle, cost_emg=cost_emg,
               cost_total=cost_settle + cost_emg)
    tot = float(out["cost_total"][REP].sum())
    log(f"  #1 P3提前调整(T时): {tot:,.0f} (结算 {float(out['cost_settle'][REP].sum()):,.0f}, "
        f"紧急 {float(out['cost_emg'][REP].sum()):,.0f}), {time.time()-t0:.0f}s")
    return out


# ================================================================
# #3: 0:00/0:10 偏移量化——比较日初 SOC 在 0:00 vs 0:10 的差异
# ================================================================
def quantify_010_offset():
    """量化 [0:00, 0:10) 槽对日初 SOC 的影响。"""
    sim = np.load(RESULTS / "q2_main_sim.npz")
    c, d = sim["c"], sim["d"]
    soc_start = sim["soc_start"]
    # 日初 SOC 差异 = 0:10 时 SOC - 0:00 时 SOC = [0:00,0:10) 槽的 SOC 变化
    # [0:00,0:10) 是前一日计划日 slot 143
    deltas = np.zeros(DAYS)
    for d_idx in range(1, DAYS):
        # 前一日 slot 143 的充放电
        ch_prev = c[d_idx - 1, 143]
        dis_prev = d[d_idx - 1, 143]
        deltas[d_idx] = ETA * ch_prev - dis_prev / ETA
    rep = deltas[31:]
    print(f"  #3 0:00/0:10 SOC偏移: 均值 {rep.mean():.2f} kWh, 标准差 {rep.std():.2f}, "
          f"最大 |Δ| {np.abs(rep).max():.2f} kWh")
    # 量化对费用的影响：平均每 kWh SOC 差异 × 平均电价 ≈ 影响上界
    avg_price = PRICE_A1.mean()
    cost_impact = np.abs(rep).mean() * avg_price * 2  # 充放往返损耗
    print(f"  #3 费用影响上界: {cost_impact:.2f} 元/日 × 334 日 ≈ {cost_impact*334:.0f} 元/年")
    return dict(mean_delta=float(rep.mean()), std_delta=float(rep.std()),
                max_abs=float(np.abs(rep).max()), annual_impact=float(cost_impact * 334))


# ================================================================
# 主函数
# ================================================================
def _run_job(tag):
    if tag == "no_emerg_charge":
        return tag, simulate_no_emerg_charge()
    elif tag == "p3_early":
        return tag, simulate_p3_early_adjust()
    elif tag == "offset":
        return tag, quantify_010_offset()


if __name__ == "__main__":
    offset_result = quantify_010_offset()

    results = {}
    # 逐个执行（不让一个失败炸全部）
    for tag in ["no_emerg_charge", "p3_early"]:
        try:
            results[tag] = _run_job(tag)[1]
        except Exception as e:
            print(f"  {tag} FAILED: {e}")
            results[tag] = None

    print("\n" + "=" * 60)
    base_p2 = 13879046
    base_p3 = 13475997
    for tag, r in results.items():
        if hasattr(r, "get") and "cost_total" in r:
            tot = float(r["cost_total"][REP].sum())
            if "no_emerg" in tag:
                print(f"#6 禁止紧急充电: {tot:,.0f} vs 主策略 {base_p2:,.0f} → 差 {tot - base_p2:+,.0f}")
            elif "p3_early" in tag:
                print(f"#1 P3提前调整: {tot:,.0f} vs 当前P3 {base_p3:,.0f} → 差 {tot - base_p3:+,.0f}")
        elif isinstance(r, dict):
            print(f"#3 偏移量化: {r}")

    import json
    (RESULTS / "review_rerun.json").write_text(json.dumps({
        "no_emerg_charge": float(results["no_emerg_charge"]["cost_total"][REP].sum()),
        "p3_early": float(results["p3_early"]["cost_total"][REP].sum()),
        "offset": offset_result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n结果已保存 results/review_rerun.json")
