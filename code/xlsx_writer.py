# -*- coding: utf-8 -*-
"""结果工作簿写出公共模块：计划/充放电/紧急购电三个 sheet 的统一填写。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from common import load_all, N, ETA, SOC_INIT, DAYS
from q2_simulate import clock_day_view, emergency_intervals

BLOCKS = [(0, 24, "0:00-4:00"), (24, 48, "4:00-8:00"), (48, 72, "8:00-12:00"),
          (72, 96, "12:00-16:00"), (96, 120, "16:00-20:00"), (120, 144, "20:00-24:00")]
REP_DATES = pd.date_range("2025-01-01", periods=DAYS)[31:365]


def soc_boundaries(C, D_, soc_start):
    """钟表日 0:00 / 24:00 储电量。SOC(D 0:00) = 计划日D起始SOC − 计划日D−1末槽变动。"""
    soc_0 = np.zeros(DAYS); soc_24 = np.zeros(DAYS)
    soc_0[0] = SOC_INIT
    for D in range(1, DAYS):
        soc_0[D] = soc_start[D] - (ETA * C[D - 1, 143] - D_[D - 1, 143] / ETA)
    soc_24 = np.roll(soc_0, -1)
    soc_24[364] = soc_start[364] + float(np.sum(ETA * C[364, :143] - D_[364, :143] / ETA))
    return soc_0, soc_24


def write_plan_sheet(ws, g_plan, cost_col, dates=REP_DATES):
    """计划购电量/调整购电量 sheet：144 槽 + 全天购电量 + 全天购电费（cost_col 为对应费用）。"""
    hdr = [ws.cell(row=1, column=c).value for c in range(1, 148)]
    assert hdr[1] == "0:10-0:20" and hdr[144] == "0:00-0:10+1", f"列标签异常: {hdr[1]},{hdr[144]}"
    ws.delete_rows(2, ws.max_row)
    for i, D in enumerate(range(31, 365)):
        r = i + 2
        ws.cell(row=r, column=1, value=dates[i].to_pydatetime())
        for k in range(N):
            ws.cell(row=r, column=2 + k, value=round(float(g_plan[D, k]), 4))
        ws.cell(row=r, column=146, value=round(float(g_plan[D].sum()), 2))
        ws.cell(row=r, column=147, value=round(float(cost_col[D]), 2))


def write_battery_sheet(ws, C, D_, soc_0, soc_24, dates=REP_DATES):
    c_ck = clock_day_view(C); d_ck = clock_day_view(D_)
    ws.delete_rows(2, ws.max_row)
    r = 2
    for i, D in enumerate(range(31, 365)):
        for b, (a, bnd, lab) in enumerate(BLOCKS):
            ws.cell(row=r, column=1, value=dates[i].to_pydatetime() if b == 0 else None)
            ws.cell(row=r, column=2, value=lab)
            ws.cell(row=r, column=3, value=round(float(c_ck[D, a:bnd].sum()), 4))
            ws.cell(row=r, column=4, value=round(float(d_ck[D, a:bnd].sum()), 4))
            if b == 0:
                ws.cell(row=r, column=5, value="0:00")
                ws.cell(row=r, column=6, value=round(float(soc_0[D]), 2))
            elif b == 1:
                ws.cell(row=r, column=5, value="24:00")
                ws.cell(row=r, column=6, value=round(float(soc_24[D]), 2))
            r += 1


def write_emergency_sheet(ws, emg, dates=REP_DATES):
    emg_ck = clock_day_view(emg)
    ws.delete_rows(2, ws.max_row)
    r = 2
    n_days = 0
    for D in range(31, 365):
        ivs = emergency_intervals(emg_ck, D)
        if not ivs:
            continue
        n_days += 1
        for j, (lab, q) in enumerate(ivs):
            ws.cell(row=r, column=1, value=dates[D - 31].to_pydatetime() if j == 0 else None)
            ws.cell(row=r, column=2, value=lab)
            ws.cell(row=r, column=3, value=round(q, 2))
            r += 1
    return n_days


def spec_date_detail(g_plan, a_final, C, D_, EM, soc_start, cost_by_day):
    """四指定日明细（论文表用）。cost_by_day: dict(日期索引 -> 费用分解)。"""
    c_ck = clock_day_view(C); d_ck = clock_day_view(D_); emg_ck = clock_day_view(EM)
    soc_0, soc_24 = soc_boundaries(C, D_, soc_start)
    detail = {}
    for s in ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]:
        D = (pd.Timestamp(s) - pd.Timestamp("2025-01-01")).days
        t1 = {f"{(k + 1) * 10 // 60:02d}:{(k + 1) * 10 % 60:02d}":
              round(float((g_plan if a_final is None else a_final)[D, k]), 2)
              for k in [59, 71, 83, 95, 107, 119]}
        t2 = {lab: [round(float(c_ck[D, a:bnd].sum()), 2), round(float(d_ck[D, a:bnd].sum()), 2)]
              for a, bnd, lab in BLOCKS}
        detail[s] = dict(day_index=D, t1_slots=t1, t2_blocks=t2,
                         t3_emergency=emergency_intervals(emg_ck, D),
                         soc_0=round(float(soc_0[D]), 2), soc_24=round(float(soc_24[D]), 2),
                         **cost_by_day(D))
    return detail
