# -*- coding: utf-8 -*-
"""问题4：波动电价（附件4）下重新计算问题2与问题3。

信息结构：附件4 电价按日前已知处理（日前市场出清价提前发布的惯例；若视为实时未知，
则需引入电价预测模型，本文在模型评价中讨论该扩展）。负载/光伏预测、缓冲与执行
机制与问题二/三完全一致，仅电价路径替换为附件4。
退化检验：将附件4换回附件1固定电价，本脚本退化为问题二/三的原模型（代码路径一致）。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import load_all, key_numbers, log, RESULTS, N, DT, DAYS
from q2_simulate import simulate, REP
from q3_simulate import build_profiles, build_buffers, simulate_q3

A1, LOAD, PV, PRICE4, FC3 = load_all()


def _run_saa(job):
    S, load_fc, pv_fc = job
    from q2_saa import simulate_saa
    return S, simulate_saa(load_fc, pv_fc, PRICE4, S=S, exec_mode="rolling")


def _run_q3(job):
    at, lf3, pv0, pvT, buf0, bufT, errT = job
    return simulate_q3(lf3, pv0, pvT, buf0, bufT, at, exec_mode="rolling",
                       price_days=PRICE4, plan_mode="saa", errT=errT)


def main():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    load_fc, pv_fc = fc["load_fc"], fc["pv_fc"]

    log("【P4-2】问题二模型（SAA 计划层）× 波动电价")
    lf3, pv0, pvT, _, _ = build_profiles()
    buf0, bufT = build_buffers(lf3, pv0, pvT)
    from q3_simulate import build_errT
    errT = build_errT(lf3, pvT)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {S: ex.submit(_run_saa, (S, load_fc, pv_fc)) for S in [10, 14]}
        futs["main"] = ex.submit(_run_q3, ([6, 12, 18], lf3, pv0, pvT, buf0, bufT, errT))
        futs["noAdj"] = ex.submit(_run_q3, ([], lf3, pv0, pvT, buf0, bufT, errT))
        grid = {S: float(futs[S].result()[1]["cost_total"][REP].sum()) for S in [10, 14]}
        q4_2 = futs[10].result()[1]
        q4_3 = futs["main"].result()
        q4_3_noAdj = futs["noAdj"].result()

    # ---- 波动电价下的基准 ----
    base4 = float(sum(PRICE4[d] @ np.maximum(LOAD[d] * DT - PV[d] * DT, 0)
                      for d in range(31, 365)))
    oracle4 = simulate(LOAD, PV, PRICE4, buffer_q=None, exec_mode="greedy")

    print("\n波动电价（附件4）回测期对比：")
    print(f"  无储能基准        {base4:>14,.0f} 元")
    print(f"  P4-2（问题二模型） {q4_2['cost_total'][REP].sum():>14,.0f} 元 "
          f"(紧急 {q4_2['cost_emg'][REP].sum():,.0f})")
    print(f"  P4-3 不调整       {q4_3_noAdj['cost_total'][REP].sum():>14,.0f} 元")
    print(f"  P4-3 全调整(主)   {q4_3['cost_total'][REP].sum():>14,.0f} 元 "
          f"(结算 {q4_3['cost_settle'][REP].sum():,.0f}, 紧急 {q4_3['cost_emg'][REP].sum():,.0f})")
    print(f"  理想先知          {oracle4['cost_total'][REP].sum():>14,.0f} 元")

    np.savez(RESULTS / "q4_2_sim.npz",
             **{k: q4_2[k] for k in ["g_plan", "c", "d", "spill", "emg",
                                     "soc_start", "cost_plan", "cost_emg"]})
    np.savez(RESULTS / "q4_3_sim.npz",
             **{k: q4_3[k] for k in ["g_plan", "a_final", "c", "d", "spill",
                                     "emg", "soc_start", "cost_settle", "cost_emg"]})
    key_numbers(
        q4_cost_baseline=round(base4, 0),
        q4_2_cost_main=round(float(q4_2["cost_total"][REP].sum()), 0),
        q4_2_cost_emg=round(float(q4_2["cost_emg"][REP].sum()), 0),
        q4_2_cost_oracle=round(float(oracle4["cost_total"][REP].sum()), 0),
        q4_3_cost_noAdj=round(float(q4_3_noAdj["cost_total"][REP].sum()), 0),
        q4_3_cost_main=round(float(q4_3["cost_total"][REP].sum()), 0),
        q4_3_main_settle=round(float(q4_3["cost_settle"][REP].sum()), 0),
        q4_3_main_emg=round(float(q4_3["cost_emg"][REP].sum()), 0),
        q4_3_emg_total_kwh=round(float(q4_3["emg"][REP].sum()), 0),
        q4_2_sens_S={str(k): round(v, 0) for k, v in grid.items()},
        q4_price_range=[round(float(PRICE4.min()), 4), round(float(PRICE4.max()), 4)],
    )
    log("P4 仿真与关键数值已保存")


if __name__ == "__main__":
    main()
