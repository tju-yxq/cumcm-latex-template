# -*- coding: utf-8 -*-
"""一次性准备画图数据：fig_q2cmp 三策略费用分解 + 新版 SAA 灵敏度网格，
全部写入 results/key_numbers.json（论文单一数据源），画图脚本不再含硬编码或重计算。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import key_numbers, log
from q2_simulate import simulate, PRICE_A1, REP

fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
lf, pf = fc["load_fc"], fc["pv_fc"]

out = {}

log("朴素预测+贪心 ...")
s = simulate(lf, pf, PRICE_A1[None, :], buffer_q=None, exec_mode="greedy")
out["q2_cmp_naive_plan"] = float(s["cost_plan"][REP].sum())
out["q2_cmp_naive_emg"] = float(s["cost_emg"][REP].sum())
log(f"  plan={out['q2_cmp_naive_plan']:,.0f} emg={out['q2_cmp_naive_emg']:,.0f} "
    f"total={out['q2_cmp_naive_plan']+out['q2_cmp_naive_emg']:,.0f} (期望 15,402,753)")

log("报童缓冲 q=0.7 + 贪心 ...")
s = simulate(lf, pf, PRICE_A1[None, :], buffer_q=0.7, exec_mode="greedy")
out["q2_cmp_greedy_plan"] = float(s["cost_plan"][REP].sum())
out["q2_cmp_greedy_emg"] = float(s["cost_emg"][REP].sum())
log(f"  plan={out['q2_cmp_greedy_plan']:,.0f} emg={out['q2_cmp_greedy_emg']:,.0f} "
    f"total={out['q2_cmp_greedy_plan']+out['q2_cmp_greedy_emg']:,.0f} (期望 14,129,870)")

log("报童缓冲 q=0.7 + 滚动 ...")
s = simulate(lf, pf, PRICE_A1[None, :], buffer_q=0.7, exec_mode="rolling")
out["q2_cmp_bufroll_plan"] = float(s["cost_plan"][REP].sum())
out["q2_cmp_bufroll_emg"] = float(s["cost_emg"][REP].sum())
log(f"  plan={out['q2_cmp_bufroll_plan']:,.0f} emg={out['q2_cmp_bufroll_emg']:,.0f} "
    f"total={out['q2_cmp_bufroll_plan']+out['q2_cmp_bufroll_emg']:,.0f} (期望 14,039,965)")

# ---- 新版灵敏度（SAA+滚动，服务器 sens2.log 实测值，2026-09-12 13:04-13:51）----
out["sens_p2_saa_eta"] = {"0.85": 14426075.0, "0.9": 13879046.0, "0.95": 13383114.0}
out["sens_p2_saa_soc"] = {
    "(1200, 10800)": 13879046.0, "(960, 10800)": 13823584.0,
    "(1440, 10800)": 13949359.0, "(1200, 8640)": 14547002.0,
    "(1200, 12000)": 13666578.0,
}
out["sens_p2_saa_emerg"] = {
    "3": 13467684.0, "4": 13673365.0, "5": 13879046.0,
    "6": 14084728.0, "8": 14496090.0,
}
out["sens_p2_saa_emerg_plan"] = 12850641.0
# 审稿重跑结论（论文灵敏度章节 (4)(5)(6)）
out["rerun_no_emerg_charge"] = 14032132.0
out["rerun_p3_early_adjust"] = 13504099.0
out["rerun_0010_soc_offset_mean"] = 441.03
out["rerun_0010_soc_offset_std"] = 298.81

key_numbers(**out)
log("已写入 key_numbers.json")
