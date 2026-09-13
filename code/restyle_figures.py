# -*- coding: utf-8 -*-
"""论文图表管线 v3 —— 遵循 scipilot-figure-skill 规范：
- 无图内总标题（LaTeX 图注即标题，避免双重图注）
- 禁用双 Y 轴：电价与电量拆为共享 x 轴的上下子图
- 按版心实际尺寸出图（全宽 6.3in），正文/刻度 8-9pt，不二次缩放
- Okabe-Ito 色盲安全色板 + 线型冗余编码
- 每图过 visual_qa.audit_layout 程序自检（缺字/裁切/刻度重叠），FAIL 即退出
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"C:\Users\26517\Documents\CUMCM\.skill\scipilot-figure-skill\scripts")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, solve_plan, RESULTS, FIGURES, N, DT, ETA, SOC_INIT
from visual_qa import audit_layout, print_report

A1, LOAD, PV, PRICE4, FC3 = load_all()
KN = json.loads((RESULTS / "key_numbers.json").read_text(encoding="utf-8"))
T = np.arange(N) * 10 / 60.0          # 槽起点（小时）
W = 10 / 60.0                          # 槽宽（小时）

# Okabe-Ito 色盲安全色板
C = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F",
     "dark": "#2C3E50", "black": "#000000"}

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#BBBBBB", "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.5,
})
FW = 6.3        # 全宽（≈0.98\textwidth）


def panel(ax, s):
    ax.text(0.0, 1.03, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom")


def audit(fig, name):
    issues = audit_layout(fig)
    fails = [i for i in issues if i[0] == "FAIL"]
    print(f"  [audit] {name}: {len(issues)} 项"
          + ("" if not issues else f"（FAIL×{len(fails)}）"))
    if issues:
        print_report(issues)
    assert not fails, f"{name} 自检 FAIL，中止"
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def hour_axis(ax):
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻（h）")


# ============ 1. 问题一：三面板（电价/边际成本 → 调度 → SOC） ============
def fig_q1():
    price_c = np.roll(A1["price"], 1)
    load_c = np.roll(A1["load"], 1)
    pv_c = np.roll(A1["pv"], 1)
    sol = solve_plan(price_c, load_c, pv_c, soc0=SOC_INIT, cyclic=True)
    g, c, d, s = sol["g"], sol["c"], sol["d"], sol["s"]
    mc = sol["res"].eqlin.marginals[N + 1: 2 * N + 1]

    fig, axes = plt.subplots(3, 1, figsize=(FW, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.75, 1.05],
                                          "hspace": 0.14})
    ax = axes[0]
    ax.step(T, price_c, where="post", color=C["grey"], lw=1.4, label="电价")
    ax.step(T, mc, where="post", color=C["dark"], lw=1.6, ls="--",
            label="边际供电成本（对偶）")
    ax.set_ylabel("元/kWh")
    ax.set_ylim(0, 1.55)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2)
    panel(ax, "(a)")

    ax = axes[1]
    ax.step(T, g, where="post", color=C["red"], lw=1.5, label="计划购电")
    ax.bar(T, c, width=W, align="edge", color=C["blue"], alpha=0.8, label="充电")
    ax.bar(T, -d, width=W, align="edge", color=C["green"], alpha=0.8, label="放电")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("kWh/10min")
    ym = max(g.max(), c.max(), d.max()) * 1.18 + 50      # 动态上限防截断，留图例空间
    ax.set_ylim(-ym, ym)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.2)
    panel(ax, "(b)")

    ax = axes[2]
    ax.plot(T, s, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9)
    ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.text(23.8, 1180, "电量下限", ha="right", va="top", fontsize=7.5, color=C["red"])
    ax.text(23.8, 10830, "电量上限", ha="right", va="bottom", fontsize=7.5, color=C["red"])
    ax.set_ylabel("储电量（kWh）")
    ax.set_ylim(0, 12200)
    hour_axis(ax)
    panel(ax, "(c)")
    audit(fig, "fig_q1_strategy.png")


# ============ 2. 问题二：策略费用构成（堆叠柱） ============
def fig_q2cmp():
    from q2_simulate import simulate, PRICE_A1, REP
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    s_naive = simulate(lf, pf, PRICE_A1[None, :], buffer_q=None, exec_mode="greedy")
    s_greedy = simulate(lf, pf, PRICE_A1[None, :], buffer_q=0.7, exec_mode="greedy")
    sim = np.load(RESULTS / "q2_main_sim.npz")
    entries = [
        ("无储能\n基准", KN["q2_cost_baseline"], 0.0),
        ("朴素预测\n+贪心", float(s_naive["cost_plan"][REP].sum()),
         float(s_naive["cost_emg"][REP].sum())),
        ("报童缓冲\n+贪心", float(s_greedy["cost_plan"][REP].sum()),
         float(s_greedy["cost_emg"][REP].sum())),
        ("报童缓冲\n+滚动", 13103589.0, KN["q2_cost_buffer_rolling"] - 13103589.0),
        ("光伏分位数\n裕量+滚动", KN["q2_e5_plan"], KN["q2_e5_emg"]),
        ("随机规划\n+滚动（主）", float(sim["cost_plan"][REP].sum()),
         float(sim["cost_emg"][REP].sum())),
        ("理想先知", KN["q2_cost_oracle"], 0.0),
    ]
    names = [e[0] for e in entries]
    plans = np.array([e[1] for e in entries]) / 1e4
    emgs = np.array([e[2] for e in entries]) / 1e4
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(FW, 3.7))
    ax.bar(x, plans, 0.62, color=C["blue"], label="计划购电费")
    ax.bar(x, emgs, 0.62, bottom=plans, color=C["red"], label="紧急购电费（5×）")
    top = (plans + emgs)
    for xi, p, e in zip(x, plans, emgs):
        ax.text(xi, p + e + 2.2, f"{p + e:.1f}", ha="center", fontsize=8.5,
                fontweight="bold")
        if e > 0.05 * (plans.max() + emgs.max()):
            ax.text(xi, p + e / 2, f"{e / (p + e) * 100:.0f}%", ha="center",
                    fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("回测期总费用（万元）")
    ax.set_ylim(0, top.max() * 1.16)
    ax.legend(loc="upper right")
    audit(fig, "fig_q2_strategy_cmp.png")


# ============ 3. 问题二：缓冲分位数敏感性 ============
def fig_q2sens():
    sens = KN["q2_sens_q"]
    qs = sorted(float(k) for k in sens)
    vals = np.array([sens[str(q)] for q in qs]) / 1e4
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot([q * 100 for q in qs], vals, "o-", color=C["dark"], markersize=5)
    ax.axvspan(70, 80, color=C["sky"], alpha=0.22, label="平坦最优区 $[0.7,0.8]$")
    ax.axvline(80, color=C["red"], ls="--", lw=1.0, label="报童理论值 80%")
    i = int(vals.argmin())
    ax.plot(qs[i] * 100, vals[i], "o", color=C["red"], markersize=6, zorder=5)
    ax.annotate(f"最低 {vals[i]:.1f} 万元", xy=(qs[i] * 100, vals[i]),
                xytext=(qs[i] * 100 - 13, vals[i] + 2.4),
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.9),
                fontsize=8)
    ax.set_xlabel("缓冲分位数 $q$（%）")
    ax.set_ylabel("回测期总费用（万元）")
    ax.set_ylim(vals.min() - 4, vals.max() + 6)
    ax.legend(loc="upper right")
    audit(fig, "fig_q2_sens_q.png")


# ============ 4. 问题二：最严重日三面板 ============
def fig_q2day():
    sim = np.load(RESULTS / "q2_main_sim.npz")
    g, Cc, Dd, EM = sim["g_plan"], sim["c"], sim["d"], sim["emg"]
    soc_st = sim["soc_start"]
    emg_cost = sim["cost_emg"].copy(); emg_cost[:31] = 0
    Dm = int(np.argmax(emg_cost))
    dstr = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=Dm)).strftime("%Y-%m-%d")
    price = A1["price"]

    fig, axes = plt.subplots(3, 1, figsize=(FW, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.75, 1.75, 1.05],
                                          "hspace": 0.14})
    ax = axes[0]
    ax.step(T, price, where="post", color=C["grey"], lw=1.4)
    ax.set_ylabel("电价\n（元/kWh）")
    ax.set_ylim(0, 1.55)
    panel(ax, "(a)")

    ax = axes[1]
    ax.step(T, g[Dm], where="post", color=C["red"], lw=1.5, label="计划购电")
    ax.bar(T, Cc[Dm], width=W, align="edge", color=C["blue"], alpha=0.8, label="实际充电")
    ax.bar(T, -Dd[Dm], width=W, align="edge", color=C["green"], alpha=0.8, label="实际放电")
    m = EM[Dm] > 1e-6
    ax.bar(T[m], EM[Dm][m], width=W, align="edge", color=C["purple"],
           label="紧急购电")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("kWh/10min")
    ym = max(g[Dm].max(), Cc[Dm].max(), Dd[Dm].max()) * 1.18 + 50
    ax.set_ylim(-ym, ym)
    ax.legend(loc="upper left", ncol=4, columnspacing=1.0)
    panel(ax, "(b)")

    ax = axes[2]
    soc = soc_st[Dm] + np.cumsum(ETA * Cc[Dm] - Dd[Dm] / ETA)
    ax.plot(T, soc, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9)
    ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.set_ylabel("储电量（kWh）")
    ax.set_ylim(0, 12200)
    hour_axis(ax)
    panel(ax, "(c)")
    audit(fig, "fig_q2_worst_day.png")


# ============ 5. 问题三：调整策略对比 ============
def fig_q3cmp():
    names = ["不调整", "仅 6 时", "仅 12 时", "仅 18 时", "全调整\n（主策略）"]
    vals = [KN["q3_cost_noAdj"], KN["q3_cost_adj6"], KN["q3_cost_adj12"],
            KN["q3_cost_adj18"], KN["q3_cost_main"]]
    base = vals[0]
    x = np.arange(5)
    colors = [C["grey"]] + [C["sky"]] * 3 + [C["red"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.bar(x, np.array(vals) / 1e4, 0.58, color=colors)
    ax.axhline(base / 1e4, color=C["grey"], ls="--", lw=0.9)
    for xi, v in zip(x, vals):
        lab = f"{v/1e4:.1f}" if v >= base else f"{v/1e4:.1f}\n(-{(base-v)/1e4:.1f})"
        ax.text(xi, v / 1e4 + 0.9, lab, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if xi == 4 else "normal",
                color=C["dark"] if v >= base else C["green"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("回测期总费用（万元）")
    ax.set_ylim(min(vals) / 1e4 - 5, max(vals) / 1e4 + 5)
    audit(fig, "fig_q3_adj_cmp.png")


# ============ 6. 问题三：调整示例日 ============
def fig_q3day():
    sim = np.load(RESULTS / "q3_main_sim.npz")
    G, A, Cc, Dd = sim["g_plan"], sim["a_final"], sim["c"], sim["d"]
    soc_st = sim["soc_start"]
    adj = np.abs(A - G).sum(axis=1); adj[:31] = 0
    Dm = int(np.argmax(adj))
    dstr = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=Dm)).strftime("%Y-%m-%d")

    fig, axes = plt.subplots(2, 1, figsize=(FW, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.9, 1.0], "hspace": 0.13})
    ax = axes[0]
    ax.step(T, G[Dm], where="post", color=C["red"], lw=1.5, label="0:00 计划购电")
    ax.step(T, A[Dm], where="post", color=C["purple"], lw=1.6, ls="--",
            label="调整后购电")
    ax.plot(T, PV[Dm] * DT, color=C["orange"], lw=1.3, label="光伏实际（×Δt）")
    ax.axhline(0, color="k", lw=0.5)
    ym = max(G[Dm].max(), A[Dm].max(), (PV[Dm] * DT).max()) * 1.20 + 50
    ax.set_ylim(-0.08 * ym, ym)
    for h, lab in [(6, "6 时"), (12, "12 时"), (18, "18 时")]:
        ax.axvline(h, color="#555555", ls=":", lw=0.8)
        ax.text(h + 0.15, ym * 0.93, lab, fontsize=7.5, color="#555555", va="top")
    ax.set_ylabel("kWh/10min")
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0)
    panel(ax, "(a)")

    ax = axes[1]
    soc = soc_st[Dm] + np.cumsum(ETA * Cc[Dm] - Dd[Dm] / ETA)
    ax.plot(T, soc, color=C["purple"], lw=1.9)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.9)
    ax.axhline(10800, color=C["red"], ls=":", lw=0.9)
    ax.set_ylabel("储电量（kWh）")
    ax.set_ylim(0, 12200)
    hour_axis(ax)
    panel(ax, "(b)")
    audit(fig, "fig_q3_adj_day.png")


# ============ 7. 问题四：固定 vs 波动 ============
def fig_q4():
    fixed = [KN["q2_cost_baseline"], KN["q2_cost_main"], KN["q3_cost_noAdj"],
             KN["q3_cost_main"], KN["q2_cost_oracle"]]
    fluct = [KN["q4_cost_baseline"], KN["q4_2_cost_main"], KN["q4_3_cost_noAdj"],
             KN["q4_3_cost_main"], KN["q4_2_cost_oracle"]]
    labels = ["无储能\n基准", "问题二\n模型", "问题三模型\n不调整",
              "问题三模型\n全调整（主）", "理想先知"]
    x = np.arange(5)
    w = 0.37
    fig, ax = plt.subplots(figsize=(FW, 3.9))
    ax.bar(x - w / 2, np.array(fixed) / 1e4, w, color=C["grey"],
           label="固定电价（附件1）")
    ax.bar(x + w / 2, np.array(fluct) / 1e4, w, color=C["red"],
           label="波动电价（附件4）")
    mx = max(max(fixed), max(fluct)) / 1e4
    for xi, f, v in zip(x, fixed, fluct):
        ax.text(xi - w / 2, f / 1e4 + 1.4, f"{f/1e4:.0f}", ha="center", fontsize=7.8)
        ax.text(xi + w / 2, v / 1e4 + 1.4, f"{v/1e4:.0f}", ha="center", fontsize=7.8)
        ax.text(xi, max(f, v) / 1e4 + 7.2, f"+{(v/f-1)*100:.1f}%", ha="center",
                fontsize=8, color=C["dark"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("回测期总费用（万元）")
    ax.set_ylim(0, mx * 1.20)
    ax.legend(loc="upper right")
    audit(fig, "fig_q4_cmp.png")


# ============ 8. 灵敏度三联 ============
def fig_sens():
    p1_eta = KN["sens_p1_eta"]; p2_eta = KN["sens_p2_eta"]; p2e = KN["sens_p2_emerg"]
    fig, axes = plt.subplots(1, 3, figsize=(FW, 2.55))
    ax = axes[0]
    k = sorted(float(v) for v in p1_eta)
    ax.plot([v * 100 for v in k], [p1_eta[str(v)] for v in k], "o-", color=C["dark"])
    ax.axvline(90, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("充放电效率 $\\eta$（%）")
    ax.set_ylabel("问题一日购电费（元）")
    panel(ax, "(a)")
    ax = axes[1]
    k = sorted(float(v) for v in p2_eta)
    ax.plot([v * 100 for v in k], [p2_eta[str(v)] / 1e4 for v in k], "o-",
            color=C["dark"])
    ax.axvline(90, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("充放电效率 $\\eta$（%）")
    ax.set_ylabel("问题二总费用（万元）")
    panel(ax, "(b)")
    ax = axes[2]
    ms = sorted(int(m) for m in p2e)
    fix = [p2e[str(m)][0] / 1e4 for m in ms]
    re_q = [p2e[str(m)][1] / 1e4 for m in ms]
    ax.plot(ms, fix, "s--", color=C["grey"], label="固定 $q{=}0.7$")
    ax.plot(ms, re_q, "o-", color=C["red"], label="按理论重选 $q$")
    ax.axvline(5, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("紧急购电倍数 $m$")
    ax.set_ylabel("问题二总费用（万元）")
    ax.legend(loc="upper left")
    panel(ax, "(c)")
    fig.subplots_adjust(wspace=0.42, bottom=0.16)
    audit(fig, "fig_sens.png")


if __name__ == "__main__":
    print("重做全部图表（v3，规范管线）：")
    fig_q1(); fig_q2cmp(); fig_q2sens(); fig_q2day(); fig_q3cmp(); fig_q3day(); fig_q4(); fig_sens()
    # 灰度预览（色盲可分性检查用）
    from PIL import Image
    for p in FIGURES.glob("fig_*.png"):
        if p.stem.startswith("fig_"):
            Image.open(p).convert("L").save(FIGURES / (p.stem + "_gray.png"))
    print("全部图表已按规范重做并生成灰度预览")
