# -*- coding: utf-8 -*-
"""问题2 预测模块：日前负载/光伏预测器（仅用历史信息）+ 净负荷误差分位数缓冲。

预测器候选（walk-forward，严格因果）：
  A persistence      前一日曲线
  B mean7            近 7 日均值
  C weekday4         同星期几近 4 周均值
  D ewmix            指数加权混合：近期(τ=7d) + 同星期(τ=21d)，负载 0.70/0.30、光伏 0.65/0.35
冷启动（1月1日）：用附件1典型日曲线作先验。
输出：work 缓存 npz（load_fc/pv_fc, 365×144，文件列序）+ 预测误差统计。
"""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from common import load_all, DAYS, N, CODE, log

W_LOAD, W_PV = 0.70, 0.65      # 同星期分量权重
TAU_RECENT, TAU_SAME = 7.0, 21.0


def _ew_curves(history, weights):
    """history: (n_days,144); weights: (n_days,) — 返回加权平均曲线 (144,)"""
    w = weights / weights.sum() if weights.sum() > 0 else None
    return (history * w[:, None]).sum(axis=0) if w is not None else history.mean(axis=0)


def forecast_day(method, load_hist, pv_hist, d, prior_load, prior_pv):
    """返回 (load_fc, pv_fc) 各(144,)，文件列序。history 为 0..d-1 天的 (d,144)。"""
    if d == 0:
        return prior_load.copy(), prior_pv.copy()
    lh, ph = load_hist[:d], pv_hist[:d]
    if method == "persistence":
        return lh[-1].copy(), ph[-1].copy()
    if method == "mean7":
        lo = lh[-min(7, d):].mean(axis=0)
        pv = ph[-min(7, d):].mean(axis=0)
        return lo, pv
    if method == "weekday4":
        idx = np.array([i for i in range(d) if i % 7 == d % 7][-4:], dtype=int)
        if len(idx) == 0:                      # 前 7 天无同星期历史 → 近期均值回退
            k = min(4, d)
            return lh[-k:].mean(axis=0), ph[-k:].mean(axis=0)
        return lh[idx].mean(axis=0), ph[idx].mean(axis=0)
    if method == "hybrid":     # 主预测器：负载=同星期近3周加权(3:2:1)，光伏=近7日加权
        ix = [j for j in range(d - 7, d - 22, -7) if j >= 0]     # d-7,d-14,d-21
        if ix:
            w = np.arange(len(ix), 0, -1, dtype=float)
            lo = w @ lh[ix] / w.sum()
        else:
            lo = lh[-min(3, d):].mean(axis=0)
        k = min(7, d)
        wgt = np.arange(1, k + 1, dtype=float)
        pvf = wgt @ ph[d - k:d] / wgt.sum()
        return lo, pvf
    if method == "hybrid1":    # 旧版主预测器（简单均值），保留作对照
        idx = np.array([i for i in range(d) if i % 7 == d % 7][-4:], dtype=int)
        lo = lh[idx].mean(axis=0) if len(idx) else lh[-min(4, d):].mean(axis=0)
        pvf = ph[-min(7, d):].mean(axis=0)
        return lo, pvf
    if method == "ewmix":
        ages = np.arange(d - 1, -1, -1)                     # 最近为 0
        w_recent = np.exp(-ages / TAU_RECENT)
        same = np.array([i for i in range(d) if i % 7 == d % 7])
        rec_l = _ew_curves(lh, w_recent); rec_p = _ew_curves(ph, w_recent)
        if len(same) > 0:
            w_same = np.exp(-(d - 1 - same) / TAU_SAME)
            sam_l = _ew_curves(lh[same], w_same); sam_p = _ew_curves(ph[same], w_same)
        else:
            sam_l, sam_p = rec_l, rec_p
        return W_LOAD * sam_l + (1 - W_LOAD) * rec_l, W_PV * sam_p + (1 - W_PV) * rec_p
    raise ValueError(method)


def run_all():
    a1, load, pv, price4, fc3 = load_all()
    prior_load, prior_pv = a1["load"], a1["pv"]
    methods = ["persistence", "mean7", "weekday4", "ewmix", "hybrid1", "hybrid"]
    fc = {m: {"load": np.zeros((DAYS, N)), "pv": np.zeros((DAYS, N))} for m in methods}
    for m in methods:
        for d in range(DAYS):
            lf, pf = forecast_day(m, load, pv, d, prior_load, prior_pv)
            fc[m]["load"][d], fc[m]["pv"][d] = lf, pf

    # 误差统计（2月-12月，334天正式期；1月为过渡期单列）
    rep = slice(31, 365)
    print(f"{'方法':<12}{'负载MAE':>10}{'负载RMSE':>10}{'光伏MAE':>10}{'净负荷MAE':>12}")
    for m in methods:
        le = fc[m]["load"][rep] - load[rep]
        pe = fc[m]["pv"][rep] - pv[rep]
        ne = (fc[m]["load"][rep] - fc[m]["pv"][rep]) - (load[rep] - pv[rep])
        print(f"{m:<12}{np.abs(le).mean():>10.1f}{np.sqrt((le**2).mean()):>10.1f}"
              f"{np.abs(pe).mean():>10.1f}{np.abs(ne).mean():>12.1f}")

    # 保存主预测器（hybrid：加权同星期3周负载 + 近7日加权光伏）与全部候选
    np.savez(CODE / "cache" / "forecast_p2.npz",
             load_fc=fc["hybrid"]["load"], pv_fc=fc["hybrid"]["pv"],
             **{f"{m}_load": fc[m]["load"] for m in methods},
             **{f"{m}_pv": fc[m]["pv"] for m in methods})
    # 兼容别名（旧代码引用 mean7_pv / hybrid2_*）
    z = dict(np.load(CODE / "cache" / "forecast_p2.npz"))
    z["hybrid2_load"], z["hybrid2_pv"] = fc["hybrid"]["load"], fc["hybrid"]["pv"]
    np.savez(CODE / "cache" / "forecast_p2.npz", **z)
    log("预测缓存已保存 cache/forecast_p2.npz")

    # 星期效应强度（论文 EDA 素材）
    dow = pd.date_range("2025-01-01", periods=DAYS).dayofweek.to_numpy()
    mean_curve = load.mean(axis=0)
    var_all = load.var(axis=0).mean()
    var_res = 0.0
    for k in range(7):
        grp = load[dow == k]
        var_res += len(grp) / DAYS * ((grp - grp.mean(axis=0)) ** 2).mean(axis=0).mean()
    print(f"\n负载星期效应: 总方差 {var_all:.0f} -> 去星期均值后 {var_res:.0f} (解释 {(1-var_res/var_all)*100:.1f}%)")
    mon = pd.date_range("2025-01-01", periods=DAYS).month.to_numpy()
    pv_monthly = np.array([pv[mon == m].sum() / 6 / (mon == m).sum() for m in range(1, 13)])
    print("光伏日均电量(万kWh/天) 按月:", np.round(pv_monthly / 1e4, 2))


if __name__ == "__main__":
    run_all()
