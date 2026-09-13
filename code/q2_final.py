# -*- coding: utf-8 -*-
"""hybrid2 + 执行层日内自适应修正 + S 网格（最后一轮冲榜实验）。

自适应执行：槽 k 的滚动 LP 中，剩余日预测不再用 0:00 静态预测，而是
- 光伏：乘性修正 ratio = 今日已观测光伏 / 预测光伏（clip [0.5,1.5]）
- 负载：加性修正 delta = 今日已观测负载偏差均值（clip ±300 kW）
只用当日已发生观测，严格因果。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import load_all, log, RESULTS, N, DT, DAYS
from q2_saa import solve_saa_plan
from q2_simulate import exec_lp, PRICE_A1, REP

A1, LOAD, PV, PRICE4, FC3 = load_all()
E_MAX = 5000 * DT
fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf, pf = fc["hybrid2_load"], fc["hybrid2_pv"]
err_all = (lf - pf) - (LOAD - PV)


def simulate(S=10, adaptive=True):
    g_plan = np.zeros((DAYS, N)); C = np.zeros_like(g_plan); D_ = np.zeros_like(g_plan)
    SP = np.zeros_like(g_plan); EM = np.zeros_like(g_plan)
    soc_start = np.zeros(DAYS); cost_plan = np.zeros(DAYS); cost_emg = np.zeros(DAYS)
    soc = 6000.0
    t0 = time.time()
    for d in range(DAYS):
        soc_start[d] = soc
        errs = err_all[max(0, d - S):d] if d else np.zeros((0, N))
        if len(errs):
            g = solve_saa_plan(PRICE_A1, lf[d], pf[d], errs, soc, S=len(errs))
        else:
            from common import solve_plan
            g = solve_plan(PRICE_A1, lf[d], pf[d], soc0=soc, cyclic=True,
                           validate=False)["g"]
        g_plan[d] = g
        cost_plan[d] = float(PRICE_A1 @ g)
        net = g + PV[d] * DT - LOAD[d] * DT
        for k in range(N):
            nk = net[k]
            if nk >= 0:
                ch = min(nk, E_MAX, (10800 - soc) / 0.9)
                dis, em, sp = 0.0, 0.0, nk - ch
            else:
                deficit = -nk
                if soc > 1200 + 1e-6:
                    if adaptive and k >= 6:
                        obs_pv = float(PV[d, :k].sum())
                        fc_pv = float(pf[d, :k].sum())
                        ratio = np.clip(obs_pv / max(fc_pv, 1.0), 0.5, 1.5)
                        delta = np.clip((LOAD[d, :k] - lf[d, :k]).mean(), -300.0, 300.0)
                        pv_rem = np.concatenate([[PV[d, k]], pf[d, k + 1:] * ratio])
                        load_rem = np.concatenate([[LOAD[d, k]], lf[d, k + 1:] + delta])
                    else:
                        pv_rem = np.concatenate([[PV[d, k]], pf[d, k + 1:]])
                        load_rem = np.concatenate([[LOAD[d, k]], lf[d, k + 1:]])
                    ch, dis, em, sp = exec_lp(PRICE_A1[k:], load_rem, pv_rem, g[k:],
                                               soc, soc_start[d])
                else:
                    dis = min(deficit, E_MAX, (soc - 1200) * 0.9)
                    em = deficit - dis; ch, sp = 0.0, 0.0
            C[d, k], D_[d, k], SP[d, k], EM[d, k] = ch, dis, sp, em
            soc += 0.9 * ch - dis / 0.9
        cost_emg[d] = float(PRICE_A1 @ EM[d]) * 5.0
    out = dict(g_plan=g_plan, c=C, d=D_, spill=SP, emg=EM, soc_start=soc_start,
               cost_plan=cost_plan, cost_emg=cost_emg, cost_total=cost_plan + cost_emg)
    log(f"  hybrid2[S={S},adp={int(adaptive)}]: 回测期 {out['cost_total'][REP].sum():,.0f} "
        f"(计划 {out['cost_plan'][REP].sum():,.0f}, 紧急 {out['cost_emg'][REP].sum():,.0f}), "
        f"{time.time() - t0:.1f}s")
    return out


def _run(job):
    S, adp = job
    return f"S{S}_adp{int(adp)}", simulate(S=S, adaptive=adp)


def main():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(10, True), (10, False), (8, True), (12, True)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(_run, jobs))
    print("\n===== 最后一轮（hybrid2 基线 13,879,930；Ma-ya6 门槛 13,834,487）=====")
    best_tag, best = None, np.inf
    for tag, r in sorted(results.items()):
        tot = float(r["cost_total"][REP].sum())
        print(f"{tag:<12}{tot:>15,.0f}")
        if tot < best:
            best_tag, best = tag, tot
    print(f"\n最优 {best_tag} = {best:,.0f}  "
          f"{'✓✓ 跨过 Ma-ya6' if best < 13834487 else '✗ 仍未过'}")
    np.savez(RESULTS / "q2_final_sim.npz",
             **{k: results[best_tag][k] for k in
                ["g_plan", "c", "d", "spill", "emg", "soc_start", "cost_plan", "cost_emg"]})


if __name__ == "__main__":
    main()
