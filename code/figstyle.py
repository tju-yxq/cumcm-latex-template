# -*- coding: utf-8 -*-
"""论文图表统一风格 v4（Okabe-Ito 色板，按版心实际尺寸出图，程序自检闭环）。

规范要点：
- 无图内总标题（LaTeX 图注即标题）；子图编号不烧入图片，由图注按阅读顺序描述
- 禁用双 Y 轴；柱状图纵轴一律从零开始；不放 "+x%" 式简陋标注
- 图例优先放面板上方外侧（legend_above），杜绝压数据
- 每张图过 visual_qa.audit_layout 自检（缺字=FAIL 即中止；裁切/重叠=WARN 打印）
- 宽度 = 论文 \\textwidth(16cm=6.299in) × includegraphics 的 width 比例，印刷 1:1 不缩放
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\26517\Documents\CUMCM\.skill\scipilot-figure-skill\scripts")
from visual_qa import audit_layout, print_report  # noqa: E402

TEXTW = 6.299  # \textwidth = 21cm - 2*2.5cm = 16cm = 6.299in

C = {  # Okabe-Ito 色盲安全色板
    "red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
    "grey": "#7F7F7F", "dark": "#2C3E50",
}


def tw(frac):
    """includegraphics width 比例 → 设计宽度（英寸），字体大小即印刷大小。"""
    return TEXTW * frac


def setup(small=False):
    """统一 rcParams。small=True 用于 0.48 版心以下的小图。"""
    base = 9.0 if small else 9.5
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": 300,
        "font.size": base,
        "axes.labelsize": base + 0.5, "axes.titlesize": base + 0.5,
        "xtick.labelsize": base - 1.0, "ytick.labelsize": base - 1.0,
        "legend.fontsize": base - 1.0, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#BBBBBB", "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.6,
        "figure.constrained_layout.use": False,
    })


def panel_label(ax, s):
    """子图标签不画在图上——由 LaTeX 图注按阅读顺序描述各面板。"""
    pass


def legend_above(ax, ncol=None, fontsize=None, x=0.0):
    """图例放面板上方外侧（无子图编号，从面板左缘开始），杜绝压数据。"""
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    if ncol is None:
        ncol = len(labels)
    ax.legend(loc="lower left", bbox_to_anchor=(x, 1.02, 1.0 - x, 0.12),
              ncol=ncol, frameon=False, fontsize=fontsize,
              columnspacing=1.1, handlelength=1.7, borderaxespad=0.0)


def hour_axis(ax):
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻（h）")


def audit_save(fig, name, figures_dir):
    """程序自检 → 保存 300dpi PNG。FAIL 即中止（不静默跳过），WARN 打印留痕。"""
    issues = audit_layout(fig)
    fails = [i for i in issues if i[0] == "FAIL"]
    if issues:
        print(f"  [audit] {name}: {len(issues)} 项（FAIL×{len(fails)}）")
        print_report(issues)
    if fails:
        plt.close(fig)
        raise SystemExit(f"{name} 自检 FAIL，中止")
    out = Path(figures_dir) / name
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    kb = out.stat().st_size // 1024
    print(f"  {name} ✓ ({kb} KB)")
