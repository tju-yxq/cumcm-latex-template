# -*- coding: utf-8 -*-
"""新增 9 张论文图表：EDA（热力图/箱线/分布/电价模式）+ 模型验证（预测overlay/全年SOC/月度费用分解/残差/调整量）。
遵循 scipilot 规范：无图内标题、Okabe-Ito 色板、按版心尺寸、visual_qa 自检。
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
import figstyle
from figstyle import C, panel_label
figstyle.setup()
from visual_qa import audit_layout, print_report
from common import load_all, RESULTS, FIGURES, N, DT, DAYS

A1, LOAD, PV, PRICE4, FC3 = load_all()
KN = json.loads((RESULTS / "key_numbers.json").read_text(encoding="utf-8"))
FW = 6.3
plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei"],
                     "axes.unicode_minus": False})

def audit_save(fig, name):
    issues = audit_layout(fig)
    fails = [i for i in issues if i[0] == "FAIL"]
    if fails:
        print(f"  [FAIL] {name}:")
        print_report(issues)
        return
    if issues:
        print(f"  [WARN] {name}: {len(issues)} items")
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} ✓")

# ============ 1. 负载热力图：周几 × 时段 ============
def fig_load_heatmap():
    dates = pd.date_range("2025-01-01", periods=DAYS)
    dow = dates.dayofweek.to_numpy()   # 0=Mon
    # 平均负载：(7, 24) 周几 × 小时
    load_hourly = LOAD.reshape(DAYS, 24, 6).mean(axis=2)  # (365, 24)
    hm = np.zeros((7, 24))
    for k in range(7):
        hm[k] = load_hourly[dow == k].mean(axis=0)
    fig, ax = plt.subplots(figsize=(FW, 3.2))
    im = ax.imshow(hm, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([f"{h}" for h in range(0, 24, 4)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
    ax.set_xlabel("时刻（h）")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("平均负载（kW）", fontsize=8)
    ax.grid(False)
    panel_label(ax, "(a)")
    audit_save(fig, "fig_eda_load_heatmap.png")

# ============ 2. 光伏月度箱线图 ============
def fig_pv_monthly():
    dates = pd.date_range("2025-01-01", periods=DAYS)
    months = dates.month.to_numpy()
    pv_daily = PV.sum(axis=1) / 6  # kWh/day
    data = [pv_daily[months == m] for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(FW, 3.2))
    bp = ax.boxplot(data, positions=range(1, 13), widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=2, alpha=0.4),
                    medianprops=dict(color=C["dark"], lw=1.5),
                    boxprops=dict(facecolor=C["sky"], alpha=0.6),
                    whiskerprops=dict(color=C["grey"]), capprops=dict(color=C["grey"]))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=7.5)
    ax.set_ylabel("日发电量（kWh）")
    ax.set_xlabel("月份")
    panel_label(ax, "(b)")
    audit_save(fig, "fig_eda_pv_monthly.png")

# ============ 3. 电价日内模式对比 ============
def fig_price_pattern():
    t = np.arange(N) * 10 / 60
    fig, ax = plt.subplots(figsize=(FW, 3.0))
    # 固定电价（附件1）
    ax.step(t, A1["price"], where="post", color=C["grey"], lw=1.8, label="固定电价（附件1）")
    # 波动电价：取 4 个代表日的均值±范围
    sample_days = [31, 120, 212, 304]  # 2/1, 5/1, 8/1, 11/1
    for i, d in enumerate(sample_days):
        ax.plot(t, PRICE4[d], color=C["red"], alpha=0.25, lw=0.6)
    mean4 = PRICE4[sample_days].mean(axis=0)
    ax.plot(t, mean4, color=C["red"], lw=1.5, label="波动电价均值（4 个代表日）")
    ax.fill_between(t, PRICE4[sample_days].min(axis=0),
                    PRICE4[sample_days].max(axis=0),
                    color=C["red"], alpha=0.12, label="波动范围")
    ax.set_xlabel("时刻（h）"); ax.set_ylabel("电价（元/kWh）")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.legend(loc="upper left", fontsize=7.5)
    panel_label(ax, "(c)")
    audit_save(fig, "fig_eda_price.png")

# ============ 4. 净负荷分布 ============
def fig_netload_dist():
    net = (LOAD - PV).flatten() / 1e3  # MW
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(net, bins=80, density=True, color=C["sky"], alpha=0.65,
            edgecolor="white", linewidth=0.3)
    from scipy import stats
    kde = stats.gaussian_kde(net)
    xs = np.linspace(net.min(), net.max(), 300)
    ax.plot(xs, kde(xs), color=C["dark"], lw=1.8, label="核密度估计")
    ax.axvline(net.mean(), color=C["red"], ls="--", lw=1.0,
               label=f"均值 {net.mean():.2f}")
    ax.set_xlabel("净负荷（MW）"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=8)
    panel_label(ax, "(d)")
    audit_save(fig, "fig_eda_netload_dist.png")

# ============ 5. 预测 vs 实际 overlay（3 个样本日） ============
def fig_forecast_overlay():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    t = np.arange(N) * 10 / 60
    sample = [("2025-03-15（周六）", 73), ("2025-06-15（周日）", 165),
              ("2025-09-15（周一）", 257)]
    fig, axes = plt.subplots(1, 3, figsize=(FW, 2.4), sharey=False)
    for i, (label, d) in enumerate(sample):
        ax = axes[i]
        ax.plot(t, LOAD[d], color=C["blue"], lw=1.2, label="实际负载")
        ax.plot(t, lf[d], color=C["red"], lw=1.2, ls="--", label="预测负载")
        ax.plot(t, PV[d], color=C["orange"], lw=1.0, label="实际光伏")
        ax.plot(t, pf[d], color=C["green"], lw=1.0, ls="--", label="预测光伏")
        ax.set_title(label, fontsize=7.5)
        ax.set_xlabel("时刻（h）", fontsize=7.5)
        ax.set_xticks(range(0, 25, 8))
        if i == 0:
            ax.set_ylabel("功率（kW）", fontsize=8)
            ax.legend(fontsize=5.5, ncol=2, loc="upper left")
        ax.tick_params(labelsize=6.5)
    fig.subplots_adjust(wspace=0.28, bottom=0.18)
    audit_save(fig, "fig_val_forecast_overlay.png")

# ============ 6. 全年 SOC 轨迹 ============
def fig_soc_year():
    sim = np.load(RESULTS / "q2_main_sim.npz")
    soc = sim["soc_start"]  # (365,) 每日起始SOC
    dates = pd.date_range("2025-01-01", periods=DAYS)
    fig, ax = plt.subplots(figsize=(FW, 2.8))
    ax.plot(dates, soc, color=C["purple"], lw=0.7)
    ax.axhline(1200, color=C["red"], ls=":", lw=0.8)
    ax.axhline(10800, color=C["red"], ls=":", lw=0.8, label="SOC 安全上下限")
    ax.fill_between(dates, 1200, 10800, color=C["purple"], alpha=0.04)
    ax.set_ylabel("储电量（kWh）")
    ax.set_xlabel("日期")
    ax.set_ylim(0, 12000)
    ax.legend(loc="lower right", fontsize=7.5)
    # 月度均值标注
    for m in range(2, 13):
        mask = dates.month == m
        if mask.sum() > 0:
            ax.hlines(soc[mask].mean(), dates[mask][0], dates[mask][-1],
                      color=C["dark"], lw=2.0, alpha=0.6)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    panel_label(ax, "(a)")
    audit_save(fig, "fig_val_soc_year.png")

# ============ 7. 月度费用分解 ============
def fig_monthly_cost():
    sim = np.load(RESULTS / "q2_main_sim.npz")
    dates = pd.date_range("2025-01-01", periods=DAYS)
    months = dates.month.to_numpy()
    plan_m = np.array([sim["cost_plan"][(months == m) & (np.arange(DAYS) >= 31)].sum()
                       for m in range(2, 13)]) / 1e4
    emg_m = np.array([sim["cost_emg"][(months == m) & (np.arange(DAYS) >= 31)].sum()
                      for m in range(2, 13)]) / 1e4
    fig, ax = plt.subplots(figsize=(FW, 3.2))
    x = np.arange(2, 13)
    ax.bar(x, plan_m, 0.65, color=C["blue"], label="计划购电费")
    ax.bar(x, emg_m, 0.65, bottom=plan_m, color=C["red"], label="紧急购电费（5×）")
    for xi, p, e in zip(x, plan_m, emg_m):
        ax.text(xi, p + e + 1.5, f"{p+e:.0f}", ha="center", fontsize=7.5,
                fontweight="bold")
        if e > 3:
            ax.text(xi, p + e / 2, f"{e/(p+e)*100:.0f}%", ha="center",
                    fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}月" for m in x], fontsize=7.5)
    ax.set_ylabel("月度费用（万元）")
    ax.set_ylim(0, max(plan_m + emg_m) * 1.18)
    ax.legend(loc="upper right", fontsize=8)
    panel_label(ax, "(b)")
    audit_save(fig, "fig_val_monthly_cost.png")

# ============ 8. 预测残差分析 ============
def fig_residual():
    fc = np.load(Path(__file__).parent / "cache" / "forecast_p2.npz")
    lf, pf = fc["load_fc"], fc["pv_fc"]
    net_fc = lf - pf
    net_act = LOAD - PV
    resid = (net_act - net_fc).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(FW, 2.8))
    # (a) 残差直方图
    ax = axes[0]
    ax.hist(resid, bins=80, density=True, color=C["sky"], alpha=0.65,
            edgecolor="white", linewidth=0.2)
    from scipy import stats
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()),
            color=C["dark"], lw=1.5, ls="--", label="正态拟合")
    ax.set_xlabel("残差（kW）"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=7.5)
    panel_label(ax, "(a)")
    # (b) QQ 图
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(resid[::10], dist="norm")
    ax.scatter(osm, osr, s=1.5, color=C["sky"], alpha=0.4)
    ax.plot(osm, slope * osm + intercept, color=C["red"], lw=1.2,
            label=f"$R^2={r**2:.3f}$")
    ax.set_xlabel("理论分位数"); ax.set_ylabel("样本分位数")
    ax.legend(fontsize=7.5)
    panel_label(ax, "(b)")
    fig.subplots_adjust(wspace=0.32, bottom=0.15)
    audit_save(fig, "fig_val_residual.png")

# ============ 9. 问题三调整量分布 ============
def fig_adj_distribution():
    sim = np.load(RESULTS / "q3_main_sim.npz")
    G, A = sim["g_plan"], sim["a_final"]
    dev = (A - G)  # (365, 144)
    dev_rep = dev[31:]  # 正式期
    fig, axes = plt.subplots(1, 2, figsize=(FW, 2.8))
    # (a) 逐日总调整量
    ax = axes[0]
    daily_adj = np.abs(dev_rep).sum(axis=1)
    ax.bar(range(len(daily_adj)), daily_adj, color=C["purple"], alpha=0.55, width=1.0)
    ax.axhline(daily_adj.mean(), color=C["dark"], ls="--", lw=1.0,
               label=f"均值 {daily_adj.mean():.0f} kWh/日")
    ax.set_xlabel("天数"); ax.set_ylabel("日调整总量（kWh）")
    ax.legend(fontsize=7.5)
    panel_label(ax, "(a)")
    # (b) 调整方向占比
    ax = axes[1]
    up = (dev_rep > 1e-6).sum()
    down = (dev_rep < -1e-6).sum()
    same = dev_rep.size - up - down
    wedges, texts, autotexts = ax.pie(
        [same, up, down], labels=["未调整", "调增", "调减"],
        colors=[C["grey"], C["red"], C["blue"]],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8},
        pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(7)
    panel_label(ax, "(b)")
    fig.subplots_adjust(wspace=0.25)
    audit_save(fig, "fig_val_adj_dist.png")


if __name__ == "__main__":
    print("生成 9 张新增图表：")
    fig_load_heatmap()
    fig_pv_monthly()
    fig_price_pattern()
    fig_netload_dist()
    fig_forecast_overlay()
    fig_soc_year()
    fig_monthly_cost()
    fig_residual()
    fig_adj_distribution()
    # 灰度预览
    from PIL import Image
    for p in FIGURES.glob("fig_eda_*.png"):
        Image.open(p).convert("L").save(FIGURES / (p.stem + "_gray.png"))
    for p in FIGURES.glob("fig_val_*.png"):
        Image.open(p).convert("L").save(FIGURES / (p.stem + "_gray.png"))
    print("全部完成，含灰度预览")
