# -*- coding: utf-8 -*-
"""
2026 CUMCM C题 公共模块：数据加载、时间约定、LP 求解、执行仿真。

时间约定（已验证，详见 .skill/data-preprocessing 与 .skill/microgrid-modeling）：
- 附件1/2/4 的 10 分钟数据行标签 t = 时段 [t, t+10min) 起点；
  144 个标签 0:10,...,23:50,0:00+1 构成"计划日"窗口 [0:10, 次日0:10)。
- 全局槽号 m = d*144 + j：d=距2025-01-01天数，j=钟表日槽号(0..143, 槽j=[j*10,(j+1)*10))。
  附件文件行 d 列 k(k=0..143) 对应全局槽 d*144+k+1；全局槽0([1月1日0:00,0:10))无数据（过渡期）。
- 附件3：发布时刻T的"预报k小时" = T+k 整点的瞬时功率(kW)。
"""
import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[2]      # .../CUMCM
ATTACH = ROOT / "C题" / "附件"
CODE = Path(__file__).resolve().parent          # .../overleaf-paper/code
RESULTS = CODE.parent / "results"
FIGURES = CODE.parent / "figures"
CACHE = CODE / "cache"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 常量
N = 144                 # 每日时段数（10 分钟）
DT = 10.0 / 60.0        # 时段长度 (h)
P_MAX = 5000.0          # 最大充/放电功率 (kW)
E_MAX = P_MAX * DT      # 单时段最大充/放电量 (kWh) = 833.33
SOC_MIN, SOC_MAX = 1200.0, 10800.0
ETA = 0.9               # 充放电效率（双向）
SOC_INIT = 6000.0       # 2025-01-01 0:00 (kWh)
EMERG_MULT = 5.0        # 紧急购电价格倍数
UP_MULT = 1.5           # 调整增购部分的电价倍数
DOWN_REFUND = 0.5       # 调整减购部分的退还比例（违约电价 = 50%）
DAYS = 365
ISSUE_HOURS = [0, 6, 12, 18]
SEED = 2026


# ---------------------------------------------------------------- 数据加载
def _label_to_minute(x):
    """列标签 → 距 0:00 的分钟数（0:00+1 → 1440）。"""
    if isinstance(x, str):
        if "+" in x:
            return 1440
        h, m = x.split(":")
        return int(h) * 60 + int(m)
    if hasattr(x, "hour"):  # datetime.time
        return x.hour * 60 + x.minute
    raise ValueError(f"未知标签: {x!r}")


def _load_wide(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    dates = pd.to_datetime(df.iloc[:, 0])
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all(), "日期不连续"
    cols = [_label_to_minute(c) for c in df.columns[1:]]
    assert cols == [(k + 1) * 10 for k in range(N)], f"列标签异常: {cols[:3]}..."
    return df.iloc[:, 1:].to_numpy(dtype=float)


def load_all(force=False):
    """返回 (a1, load, pv, price4, fc3)，带 parquet 缓存。
    a1: dict(price/load/pv 各(144,)，文件列序 0:10..0:00+1)
    load/pv/price4: (365,144) 文件列序
    fc3: (365,4,24)，fc3[d,i,k] = 第d天 ISSUE_HOURS[i] 发布的 预报(k+1)小时
    """
    c = CACHE
    if force or not (c / "load.parquet").exists():
        a1df = pd.read_excel(ATTACH / "附件1.xlsx")
        a1 = {
            "price": a1df.iloc[:, 1].to_numpy(dtype=float),
            "load": a1df.iloc[:, 2].to_numpy(dtype=float),
            "pv": a1df.iloc[:, 3].to_numpy(dtype=float),
        }
        load = _load_wide(ATTACH / "附件2.xlsx", "小区负载")
        pv = _load_wide(ATTACH / "附件2.xlsx", "光伏发电实际功率")
        price4 = _load_wide(ATTACH / "附件4.xlsx", "Sheet1")
        a3 = pd.read_excel(ATTACH / "附件3.xlsx")
        a3["日期"] = a3["日期"].ffill()
        fc3 = np.zeros((DAYS, 4, 24))
        for r in range(len(a3)):
            d = (pd.to_datetime(a3.iloc[r, 0]) - pd.Timestamp("2025-01-01")).days
            ih = int(str(a3.iloc[r, 1]).split(":")[0]) // 6  # 0/6/12/18 -> 0/1/2/3
            fc3[d, ih, :] = a3.iloc[r, 2:26].to_numpy(dtype=float)
        np.save(c / "a1.npy", a1, allow_pickle=True)
        for name, arr in [("load", load), ("pv", pv), ("price4", price4), ("fc3", fc3)]:
            np.save(c / f"{name}.npy", arr)
    else:
        a1 = np.load(c / "a1.npy", allow_pickle=True).item()
        load = np.load(c / "load.npy")
        pv = np.load(c / "pv.npy")
        price4 = np.load(c / "price4.npy")
        fc3 = np.load(c / "fc3.npy")
    return a1, load, pv, price4, fc3


# ---------------------------------------------------------------- 时间工具
def clock_slice(day, j):
    """钟表日 day 的槽 j (0..143) 对应 (文件行, 文件列)；j=0 借用前一日末列。
    返回 (row, col)，row=-1 表示全局槽0（无数据）。"""
    if j == 0:
        return (day - 1, N - 1) if day >= 1 else (-1, -1)
    return day, j - 1


def clock_day_matrix(file_arr, day):
    """把文件列序数组 (365,144) 取钟表日 day 的 144 槽 (clock j=0..143)。"""
    out = np.empty(N)
    r, c = clock_slice(day, 0)
    out[0] = file_arr[r, c] if r >= 0 else np.nan
    out[1:] = file_arr[day, :N - 1]
    return out


def to_file_order(clock_vec):
    """钟表日向量 (j=0..143) → 计划日文件列序 (0:10..0:00+1)。
    计划日第 k 列 = 时段 [(k+1)*10,(k+2)*10) = 钟表槽 k+1；k=143 → 次日槽0（周期回填）。"""
    return np.concatenate([clock_vec[1:], clock_vec[:1]])


def from_file_order(file_vec):
    """计划日文件列序 → 钟表日向量。"""
    return np.concatenate([file_vec[-1:], file_vec[:-1]])


def fc_to_slots(fc24, issue_hour):
    """附件3 一次预报（点值 = T+1h..T+24h 整点瞬时功率）→ 10 分钟槽值。
    对整点值线性插值，取槽起点处的值；返回 dict: 标签分钟(相对当日0:00, >1440 表示次日) → kW。
    覆盖标签范围 [T+60, T+1440]，其中标签 T+1440 恰为末整点。"""
    T = issue_hour * 60
    pts = {T + (k + 1) * 60: float(fc24[k]) for k in range(24)}
    out = {}
    for m in range(T + 60, T + 1441, 10):
        if m >= T + 1440:
            out[m] = pts[T + 1440]
            continue
        lo = ((m - T) // 60) * 60 + T
        hi = lo + 60
        w = (m - lo) / 60.0
        out[m] = pts[lo] * (1 - w) + pts[hi] * w
    return out


def pv_profile_day(fc24, issue_hour):
    """计划日 144 槽（文件列序，标签10..1440）的光伏预测剖面。
    issue=0: 标签 60..1430 由预报插值，标签 10..50 与 1440 为夜间置 0（全天覆盖）。
    issue>0: 仅填标签 [T+60, 1440] 段，其余为 NaN（调用方沿用旧估计）。"""
    d = fc_to_slots(fc24, issue_hour)
    prof = np.full(N, np.nan)
    if issue_hour == 0:
        prof[:] = 0.0
        for m, v in d.items():
            if m <= 1430:
                prof[m // 10 - 1] = v
    else:
        for m, v in d.items():
            if m <= 1440:
                prof[m // 10 - 1] = v
    return prof


# ---------------------------------------------------------------- 计划层 LP
def solve_plan(price, load_kw, pv_kw, soc0, cyclic=True, g_lock=None, validate=True):
    """日计划 LP（144 槽，向量顺序由调用方定义：P1 用钟表日序，P2-P4 用计划日文件列序）。

    price: (144,) 元/kWh；load_kw/pv_kw: (144,) kW
    soc0: 起始 SOC；cyclic: 末槽 SOC = soc0
    g_lock: 若给定 (144,)，则购电量固定，只优化充放/弃/紧急（滚动执行用）
    返回 dict(g,c,d,spill,emg,s,cost_plan,cost_emg,status)
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    load_e = np.asarray(load_kw, float) * DT
    pv_e = np.asarray(pv_kw, float) * DT
    # 变量: g,c,d,spill,emg,s  各144
    nv = 6 * N
    G, C, D, SP, EM, S = (slice(i * N, (i + 1) * N) for i in range(6))
    fixed_g = g_lock is not None
    if fixed_g:
        g_lock = np.asarray(g_lock, float)

    # 目标: min sum p*g + 5*p*emg   （emg 仅在 g_lock 模式有意义；计划模式 g 自由时 emg 恒 0）
    obj = np.zeros(nv)
    obj[G] = price
    obj[EM] = EMERG_MULT * price

    # 等式约束: SOC 平衡 + 日循环 + 供需平衡
    # SOC 平衡: s_t - s_{t-1} - 0.9 c_t + d_t/0.9 = 0 ；t=0: = soc0
    A_eq = lil_matrix((2 * N + 1, nv))
    b_eq = np.zeros(2 * N + 1)
    for t in range(N):
        A_eq[t, S.start + t] = 1.0
        if t > 0:
            A_eq[t, S.start + t - 1] = -1.0
        A_eq[t, C.start + t] = -ETA
        A_eq[t, D.start + t] = 1.0 / ETA
        if t == 0:
            b_eq[t] = soc0
    # 日循环: s_{143} = soc0
    if cyclic:
        A_eq[N, S.start + N - 1] = 1.0
        b_eq[N] = soc0
    # 平衡: g + pv + d + emg = load + c + spill  →  g + d + emg - c - spill = load - pv
    for t in range(N):
        r = N + 1 + t
        A_eq[r, G.start + t] = 1.0
        A_eq[r, D.start + t] = 1.0
        A_eq[r, EM.start + t] = 1.0
        A_eq[r, C.start + t] = -1.0
        A_eq[r, SP.start + t] = -1.0
        b_eq[r] = load_e[t] - pv_e[t]

    bounds = [(0, None)] * nv
    for t in range(N):
        bounds[C.start + t] = (0, E_MAX)
        bounds[D.start + t] = (0, E_MAX)
        bounds[S.start + t] = (SOC_MIN, SOC_MAX)
        if fixed_g:
            v = g_lock[t]
            bounds[G.start + t] = (v, v)

    res = linprog(obj, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": res.status, "message": res.message}
    x = res.x
    out = {
        "g": x[G], "c": x[C], "d": x[D], "spill": x[SP], "emg": x[EM], "s": x[S],
        "cost_plan": float(price @ x[G]), "cost_emg": float(price @ x[EM] * EMERG_MULT),
        "status": 0, "res": res,
    }
    # 校验
    if validate:
        s = out["s"]
        assert s.min() >= SOC_MIN - 1e-6 and s.max() <= SOC_MAX + 1e-6, "SOC 越界"
        assert np.abs(s[-1] - (soc0 if cyclic else s[-1])) < 1e-4
        soc = soc0
        for t in range(N):
            soc = soc + ETA * out["c"][t] - out["d"][t] / ETA
            assert abs(soc - s[t]) < 1e-4, "SOC 递推不一致"
    return out


# ---------------------------------------------------------------- 结果/日志
def key_numbers(**kw):
    """追加/更新关键数值到 results/key_numbers.json（论文单一数据源）。"""
    p = RESULTS / "key_numbers.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data.update(kw)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def log(msg):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}")


if __name__ == "__main__":
    a1, load, pv, price4, fc3 = load_all(force=True)
    log(f"数据加载完成: load{load.shape} pv{pv.shape} price4{price4.shape} fc3{fc3.shape}")
    log(f"附件1 电价范围 [{a1['price'].min():.4f},{a1['price'].max():.4f}]")
    log(f"附件4 电价范围 [{price4.min():.4f},{price4.max():.4f}]")
    log(f"年负载 {np.nansum(load)/6/1e4:.1f}万kWh, 年光伏 {np.nansum(pv)/6/1e4:.1f}万kWh")
