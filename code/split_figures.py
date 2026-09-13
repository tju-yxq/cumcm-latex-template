# -*- coding: utf-8 -*-
"""将所有多面板图拆分为独立子图 PNG，配合 LaTeX subfigure 使用。
每个面板单独出图、单独 caption。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"C:\Users\26517\Documents\CUMCM\.skill\scipilot-figure-skill\scripts")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.setup()
from visual_qa import audit_layout
from common import load_all, solve_plan, RESULTS, FIGURES, N, DT, ETA, SOC_INIT
import matplotlib.dates as mdates

A1, LOAD, PV, PRICE4, FC3 = load_all()
KN = json.loads((RESULTS / "key_numbers.json").read_text(encoding="utf-8"))
T = np.arange(N) * 10 / 60.0
FW = 6.3
W = 10 / 60.0
plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei"],
                     "axes.unicode_minus": False})

def save_panel(fig, name):
    issues = audit_layout(fig)
    fails = [i for i in issues if i[0] == "FAIL"]
    if fails:
        print(f"  [FAIL] {name}")
        for f in fails: print(f"    {f}")
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} ✓")

C = figstyle.C

# ============ 1. fig_q1_strategy 拆分为 3 张 ============
def split_q1():
    price_c = np.roll(A1["price"], 1); load_c = np.roll(A1["load"], 1); pv_c = np.roll(A1["pv"], 1)
    sol = solve_plan(price_c, load_c, pv_c, soc0=SOC_INIT, cyclic=True)
    g, c, d, s = sol["g"], sol["c"], sol["d"], sol["s"]
    mc = sol["res"].eqlin.marginals[N + 1: 2 * N + 1]

    # (a) 电价与边际成本
    fig, ax = plt.subplots(figsize=(FW, 2.4))
    ax.step(T, price_c, where="post", color=C["grey"], lw=1.4, label="电价")
    ax.step(T, mc, where="post", color=C["dark"], lw=1.6, ls="--", label="边际供电成本")
    ax.set_ylabel("元/kWh"); ax.set_ylim(0, 1.55); ax.legend(loc="upper left", ncol=2)
    save_panel(fig, "fig_q1_a_price.png")

    # (b) 购电与充放电
    fig, ax = plt.subplots(figsize=(FW, 2.8))
    ax.step(T, g, where="post", color=C["red"], lw=1.5, label="计划购电")
    ax.bar(T, c, width=W, align="edge", color=C["blue"], alpha=0.8, label="充电")
    ax.bar(T, -d, width=W, align="edge", color=C["green"], alpha=0.8, label="放电")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("kWh/10min")
    ym = max(g.max(), c.max(), d.max()) * 1.18 + 50
    ax.set_ylim(-ym, ym); ax.legend(loc="upper left", ncol=3)
    save_panel(fig, "fig_q1_b_dispatch.png")

    # (c) SOC
    fig, ax = plt.subplots(figsize=(FW, 2.0))
    ax.plot(T, s, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9); ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.set_ylabel("储电量（kWh）"); ax.set_ylim(0, 12200)
    ax.set_xticks(range(0, 25, 4)); ax.set_xlim(0, 24); ax.set_xlabel("时刻（h）")
    save_panel(fig, "fig_q1_c_soc.png")

# ============ 2. fig_q2_worst_day 拆分为 3 张 ============
def split_q2day():
    sim = np.load(RESULTS / "q2_main_sim.npz")
    g, Cc, Dd, EM = sim["g_plan"], sim["c"], sim["d"], sim["emg"]
    soc_st = sim["soc_start"]
    emg_cost = sim["cost_emg"].copy(); emg_cost[:31] = 0
    Dm = int(np.argmax(emg_cost))
    price = A1["price"]

    fig, ax = plt.subplots(figsize=(FW, 1.8))
    ax.step(T, price, where="post", color=C["grey"], lw=1.4)
    ax.set_ylabel("电价\n（元/kWh）"); ax.set_ylim(0, 1.55)
    save_panel(fig, f"fig_q2d_a_price.png")

    fig, ax = plt.subplots(figsize=(FW, 2.8))
    ax.step(T, g[Dm], where="post", color=C["red"], lw=1.5, label="计划购电")
    ax.bar(T, Cc[Dm], width=W, align="edge", color=C["blue"], alpha=0.8, label="实际充电")
    ax.bar(T, -Dd[Dm], width=W, align="edge", color=C["green"], alpha=0.8, label="实际放电")
    m = EM[Dm] > 1e-6
    ax.bar(T[m], EM[Dm][m], width=W, align="edge", color=C["purple"], label="紧急购电")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("kWh/10min")
    ym = max(g[Dm].max(), Cc[Dm].max(), Dd[Dm].max()) * 1.18 + 50
    ax.set_ylim(-ym, ym); ax.legend(loc="upper left", ncol=4, columnspacing=1.0)
    save_panel(fig, "fig_q2d_b_dispatch.png")

    fig, ax = plt.subplots(figsize=(FW, 2.0))
    soc = soc_st[Dm] + np.cumsum(ETA * Cc[Dm] - Dd[Dm] / ETA)
    ax.plot(T, soc, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9); ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.set_ylabel("储电量（kWh）"); ax.set_ylim(0, 12200)
    ax.set_xticks(range(0, 25, 4)); ax.set_xlim(0, 24); ax.set_xlabel("时刻（h）")
    save_panel(fig, "fig_q2d_c_soc.png")

# ============ 3. fig_q3_adj_day 拆分为 2 张 ============
def split_q3day():
    sim = np.load(RESULTS / "q3_main_sim.npz")
    G, A, Cc, Dd = sim["g_plan"], sim["a_final"], sim["c"], sim["d"]
    soc_st = sim["soc_start"]
    adj = np.abs(A - G).sum(axis=1); adj[:31] = 0
    Dm = int(np.argmax(adj))

    fig, ax = plt.subplots(figsize=(FW, 3.2))
    ax.step(T, G[Dm], where="post", color=C["red"], lw=1.5, label="0:00 计划购电")
    ax.step(T, A[Dm], where="post", color=C["purple"], lw=1.6, ls="--", label="调整后购电")
    ax.plot(T, PV[Dm] * DT, color=C["orange"], lw=1.3, label="光伏实际（×Δt）")
    ax.axhline(0, color="k", lw=0.5)
    ym = max(G[Dm].max(), A[Dm].max(), (PV[Dm] * DT).max()) * 1.20 + 50
    ax.set_ylim(-0.08 * ym, ym)
    for h, lab in [(6, "6 时"), (12, "12 时"), (18, "18 时")]:
        ax.axvline(h, color="#555555", ls=":", lw=0.8)
        ax.text(h + 0.15, ym * 0.93, lab, fontsize=7.5, color="#555555", va="top")
    ax.set_ylabel("kWh/10min"); ax.legend(loc="upper left", ncol=3, columnspacing=1.0)
    save_panel(fig, "fig_q3d_a_dispatch.png")

    fig, ax = plt.subplots(figsize=(FW, 2.0))
    soc = soc_st[Dm] + np.cumsum(ETA * Cc[Dm] - Dd[Dm] / ETA)
    ax.plot(T, soc, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9); ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.set_ylabel("储电量（kWh）"); ax.set_ylim(0, 12200)
    ax.set_xticks(range(0, 25, 4)); ax.set_xlim(0, 24); ax.set_xlabel("时刻（h）")
    save_panel(fig, "fig_q3d_b_soc.png")

# ============ 4. fig_val_residual 拆分为 2 张 ============
def split_residual():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    resid = ((LOAD - PV) - (lf - pf)).flatten()
    from scipy import stats

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(resid, bins=80, density=True, color=C["sky"], alpha=0.65,
            edgecolor="white", linewidth=0.2)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()),
            color=C["dark"], lw=1.5, ls="--", label="正态拟合")
    ax.set_xlabel("残差（kW）"); ax.set_ylabel("概率密度"); ax.legend(fontsize=8)
    save_panel(fig, "fig_val_res_a_hist.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    (osm, osr), (slope, intercept, r) = stats.probplot(resid[::10], dist="norm")
    ax.scatter(osm, osr, s=1.5, color=C["sky"], alpha=0.4)
    ax.plot(osm, slope * osm + intercept, color=C["red"], lw=1.2,
            label=f"$R^2={r**2:.3f}$")
    ax.set_xlabel("理论分位数"); ax.set_ylabel("样本分位数"); ax.legend(fontsize=8)
    save_panel(fig, "fig_val_res_b_qq.png")

# ============ 5. fig_val_adj_dist 拆分为 2 张 ============
def split_adj_dist():
    sim = np.load(RESULTS / "q3_main_sim.npz")
    G, A = sim["g_plan"], sim["a_final"]
    dev = (A - G)[31:]

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    daily_adj = np.abs(dev).sum(axis=1)
    ax.bar(range(len(daily_adj)), daily_adj, color=C["purple"], alpha=0.55, width=1.0)
    ax.axhline(daily_adj.mean(), color=C["dark"], ls="--", lw=1.0,
               label=f"均值 {daily_adj.mean():.0f} kWh/日")
    ax.set_xlabel("天数"); ax.set_ylabel("日调整总量（kWh）"); ax.legend(fontsize=8)
    save_panel(fig, "fig_val_adj_a_daily.png")

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    up = (dev > 1e-6).sum(); down = (dev < -1e-6).sum(); same = dev.size - up - down
    ax.pie([same, up, down], labels=["未调整", "调增", "调减"],
           colors=[C["grey"], C["red"], C["blue"]],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    save_panel(fig, "fig_val_adj_b_pie.png")

# ============ 6. fig_sens 拆分为 3 张 ============
def split_sens():
    p1_eta = KN["sens_p1_eta"]; p2_eta = KN.get("sens_p2_eta_saa", KN.get("sens_p2_eta", {}))
    p2e = KN.get("sens_p2_emerg_saa", KN.get("sens_p2_emerg", {}))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    k = sorted(float(v) for v in p1_eta)
    ax.plot([v * 100 for v in k], [p1_eta[str(v)] for v in k], "o-", color=C["dark"])
    ax.axvline(90, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("充放电效率 $\\eta$（%）"); ax.set_ylabel("问题一购电费（元）")
    save_panel(fig, "fig_sens_a_p1eta.png")

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    k = sorted(float(v) for v in p2_eta)
    ax.plot([v * 100 for v in k], [p2_eta[str(v)] / 1e4 for v in k], "o-", color=C["dark"])
    ax.axvline(90, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("充放电效率 $\\eta$（%）"); ax.set_ylabel("问题二总费用（万元）")
    save_panel(fig, "fig_sens_b_p2eta.png")

    if p2e:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ms = sorted(int(m) for m in p2e)
        if isinstance(p2e[str(ms[0])], list):
            fix = [p2e[str(m)][0] / 1e4 for m in ms]
            ax.plot(ms, fix, "s--", color=C["grey"], label="固定计划")
        else:
            vals = [p2e[str(m)] / 1e4 for m in ms]
            ax.plot(ms, vals, "o-", color=C["red"], label="总费用")
            ax.legend(fontsize=8)
        ax.axvline(5, color="#888888", ls=":", lw=1.0)
        ax.set_xlabel("紧急购电倍数 $m$"); ax.set_ylabel("总费用（万元）")
        if not isinstance(p2e[str(ms[0])], list): ax.legend(fontsize=8)
    save_panel(fig, "fig_sens_c_emerg.png")

# ============ 7. fig_val_forecast_overlay 拆分为 3 张 ============
def split_fc_overlay():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    sample = [("2025-03-15（周六）", 73), ("2025-06-15（周日）", 165), ("2025-09-15（周一）", 257)]
    for i, (label, d) in enumerate(sample):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(T, LOAD[d], color=C["blue"], lw=1.2, label="实际负载")
        ax.plot(T, lf[d], color=C["red"], lw=1.2, ls="--", label="预测负载")
        ax.plot(T, PV[d], color=C["orange"], lw=1.0, label="实际光伏")
        ax.plot(T, pf[d], color=C["green"], lw=1.0, ls="--", label="预测光伏")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("时刻（h）", fontsize=7.5)
        ax.set_xticks(range(0, 25, 8)); ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("功率（kW）", fontsize=8)
            ax.legend(fontsize=5.5, ncol=2, loc="upper left")
        save_panel(fig, f"fig_val_fc_{chr(97+i)}.png")

if __name__ == "__main__":
    print("拆分所有多面板图为独立子图：")
    split_q1()
    split_q2day()
    split_q3day()
    split_residual()
    split_adj_dist()
    split_sens()
    split_fc_overlay()
    print("全部完成")
