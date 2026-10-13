"""
全局参数敏感性与稳健性分析（Global Sensitivity & Robustness）
=============================================================

回应核心审稿质疑："公地悲剧、产量损失区间与'HD 增产≫LD'是否依赖你手调的
参数？" 本模块用两种方法检验主要结论对参数的稳健性：

  1) 单因素扫描（One-At-a-Time, OAT）：每次只让一个参数在合理区间变动，
     其余取默认，绘制 N=100/400/800 产量损失与 N=800 龙卷风图，
     识别最关键参数。
  2) 全局 Monte Carlo：在参数空间随机采样（速率系数对数均匀、其余均匀），
     对每一组参数检验四条定性结论：
        C1 悲剧存在：ESS > s_opt（gap>0）
        C2 密度单调：gap、产量损失随密度非递减
        C3 弱敏感 HD 增益 > LD 增益（且 HD 增益为正）
        C4 非零最优：s_opt > 0（完全关闭避荫不是最优）
     并给出各密度产量损失的分布（5/50/95% 分位）。

参数区间是"合理的相对扰动"（速率系数覆盖约±一个数量级内的主要范围），
为示例敏感性设定，绝对生物学取值待公共数据（AraPheno / Rice SNP-Seek）校准。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import (GameParams, fitness, find_ess, find_population_optimum,
                   tragedy_gap, yield_loss_percentage)


# ----------------------------------------------------------------------
# 参数空间（key: (low, high, 'log'/'lin', 生物学含义)）
# ----------------------------------------------------------------------
PARAM_SPACE = {
    "k1":    (0.004, 0.024, "log", "relative-competition gain coeff."),
    "gamma": (0.05, 0.35, "lin", "absolute height gain coeff."),
    "beta":  (0.25, 0.75, "lin", "density-dependence exponent"),
    "c":     (0.20, 0.50, "log", "construction cost coeff."),
    "alpha": (1.60, 3.00, "lin", "cost curvature (>1)"),
}
DENSITIES_OAT = [100, 400, 800]
DENSITIES_MONO = [50, 100, 200, 400, 800, 1600]
N_LD, N_HD = 100, 800
G_HYPOSENS = 0.55


def yield_loss(N, p):
    return yield_loss_percentage(N, p)


def hyposensitive_gain(N, p, g=G_HYPOSENS):
    """增益 g 的弱敏感株在密度 N 相对野生型 ESS 的增产%。"""
    s_opt = find_population_optimum(N, p)
    s_ess = find_ess(N, p)
    s_g = s_opt + g * (s_ess - s_opt)
    w_ess = fitness(s_ess, s_ess, N, p)
    w_g = fitness(s_g, s_g, N, p)
    return 100.0 * (w_g - w_ess) / w_ess if w_ess > 1e-10 else 0.0


# ----------------------------------------------------------------------
# 1) 单因素扫描 OAT
# ----------------------------------------------------------------------
def oat_scan(n_points: int = 11):
    rows = []
    tornado = []
    for name, (lo, hi, scale, _) in PARAM_SPACE.items():
        if scale == "log":
            vals = np.logspace(np.log10(lo), np.log10(hi), n_points)
        else:
            vals = np.linspace(lo, hi, n_points)
        for v in vals:
            kw = {name: float(v)}
            p = GameParams(**kw)
            rec = {"param": name, "value": float(v)}
            for N in DENSITIES_OAT:
                rec[f"loss_N{N}"] = round(yield_loss(N, p), 3)
            rows.append(rec)
        # 龙卷风：低、高端相对默认（N=800）
        p_lo = GameParams(**{name: float(lo)})
        p_hi = GameParams(**{name: float(hi)})
        p_df = GameParams()
        l_lo = yield_loss(800, p_lo)
        l_hi = yield_loss(800, p_hi)
        l_df = yield_loss(800, p_df)
        tornado.append({"param": name, "low": lo, "high": hi,
                        "loss_at_low": l_lo, "loss_at_high": l_hi,
                        "loss_default": l_df,
                        "swing_low": l_lo - l_df, "swing_high": l_hi - l_df,
                        "range": abs(l_hi - l_lo)})
    df_oat = pd.DataFrame(rows)
    df_tor = pd.DataFrame(tornado).sort_values("range", ascending=True)
    return df_oat, df_tor


# ----------------------------------------------------------------------
# 2) 全局 Monte Carlo
# ----------------------------------------------------------------------
def sample_params(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    def smpl(name):
        lo, hi, scale, _ = PARAM_SPACE[name]
        if scale == "log":
            return 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
        return rng.uniform(lo, hi, n)
    keys = list(PARAM_SPACE.keys())
    arr = {k: smpl(k) for k in keys}
    return arr


def global_monte_carlo(n: int = 6000, seed: int = 42):
    arr = sample_params(n, seed)
    n_ok = 0
    cnt = {"C1_tragedy": 0, "C2_monotonic": 0, "C3_HD_gt_LD": 0,
           "C4_sopt_positive": 0, "interior_solution": 0}
    loss_grid = np.zeros((n, len(DENSITIES_MONO)))
    losses_at = {100: [], 400: [], 800: []}
    gains_LD, gains_HD = [], []
    for i in range(n):
        p = GameParams(**{k: float(arr[k][i]) for k in arr})
        # 内点解：s_opt<1 且 N=800 ESS 未触顶（排除退化边界）
        s_opt = find_population_optimum(800, p)
        ess800 = find_ess(800, p)
        interior = (0.0 < s_opt < 0.999) and (0.0 < ess800 < 0.999)
        if not interior:
            continue
        n_ok += 1
        cnt["interior_solution"] += 1

        ess100, s_opt100, gap100 = tragedy_gap(100, p)
        ess800b, _, gap800 = tragedy_gap(800, p)
        # C1 悲剧
        if gap100 > 1e-6 and gap800 > 1e-6:
            cnt["C1_tragedy"] += 1
        # C2 单调
        gaps, losses = [], []
        for j, N in enumerate(DENSITIES_MONO):
            e, o, g = tragedy_gap(N, p)
            L = yield_loss(N, p)
            gaps.append(g); losses.append(L)
            loss_grid[i, j] = L
        if np.all(np.diff(gaps) >= -1e-9) and np.all(np.diff(losses) >= -1e-9):
            cnt["C2_monotonic"] += 1
        # C3 HD > LD
        gLD = hyposensitive_gain(N_LD, p); gHD = hyposensitive_gain(N_HD, p)
        gains_LD.append(gLD); gains_HD.append(gHD)
        if gHD > 0 and gHD > gLD:
            cnt["C3_HD_gt_LD"] += 1
        # C4 s_opt>0
        if s_opt > 1e-6:
            cnt["C4_sopt_positive"] += 1
        for N in losses_at:
            losses_at[N].append(yield_loss(N, p))

    robust = {k: (100.0 * v / n_ok if n_ok else 0.0)
              for k, v in cnt.items() if k != "interior_solution"}
    quant = {}
    for N in [100, 400, 800]:
        x = np.array(losses_at[N])
        quant[N] = {"p5": np.percentile(x, 5), "p50": np.percentile(x, 50),
                    "p95": np.percentile(x, 95), "mean": np.mean(x)}
    # 密度-损失包络（仅内点行）
    valid = loss_grid[loss_grid[:, -1] > 0]
    envelope = {
        "densities": np.array(DENSITIES_MONO),
        "p5": np.percentile(valid, 5, axis=0),
        "p50": np.percentile(valid, 50, axis=0),
        "p95": np.percentile(valid, 95, axis=0),
    }
    gains = np.array(list(zip(gains_LD, gains_HD)))
    return {"n_sampled": n, "n_interior": n_ok, "robust": robust,
            "quant": quant, "envelope": envelope, "gains": gains}


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def plot_sensitivity(df_oat, df_tor, mc, fig_path: str):
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                        "grid.alpha": 0.3, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # (a) 龙卷风图：N=800 产量损失对各参数的敏感性
    ax = axes[0, 0]
    y = np.arange(len(df_tor))
    lo = df_tor["swing_low"].values
    hi = df_tor["swing_high"].values
    ax.barh(y, lo, color="#1F77B4", label="at parameter low end")
    ax.barh(y, hi, color="#D62728", label="at parameter high end")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.param}\n[{r.low:g},{r.high:g}]"
                                          for r in df_tor.itertuples()], fontsize=8)
    ax.set_xlabel("Change in yield loss at N=800 (percentage points)")
    ax.set_title("(a) One-at-a-time sensitivity (tornado)")
    ax.legend(fontsize=8, loc="lower right")

    # (b) 全局产量损失分布
    ax = axes[0, 1]
    colors = {100: "#1F77B4", 400: "#FF7F0E", 800: "#D62728"}
    for N in [100, 400, 800]:
        ax.hist(mc["quant"][N] if False else _loss_sample(mc, N), bins=40,
                density=True, alpha=0.55, color=colors[N],
                label=f"N={N} (median {mc['quant'][N]['p50']:.1f}%)")
    ax.set_xlabel("Wild-type tragedy yield loss (%)")
    ax.set_ylabel("Probability density")
    ax.set_title("(b) Global distribution of yield loss")
    ax.legend(fontsize=8)

    # (c) 密度-损失：中位 + 90% 包络
    ax = axes[1, 0]
    d = mc["envelope"]
    dens = d["densities"]
    p_df = GameParams()
    default_curve = np.array([yield_loss(N, p_df) for N in dens])
    ax.fill_between(dens, d["p5"], d["p95"], color="#D62728", alpha=0.18,
                    label="90% parameter envelope")
    ax.plot(dens, d["p50"], color="#D62728", lw=2, label="Monte Carlo median")
    ax.plot(dens, default_curve, "k--", lw=1.5, label="Default parameters")
    ax.set_xscale("log"); ax.set_xlabel("Planting density (effective neighbours), $N$")
    ax.set_ylabel("Yield loss (%)")
    ax.set_title("(c) Yield loss vs density: monotonic and bounded")
    ax.legend(fontsize=8)

    # (d) 四结论稳健率
    ax = axes[1, 1]
    labels = {"C1_tragedy": "C1 ESS > s_opt\n(tragedy exists)",
              "C2_monotonic": "C2 gap/loss rise\nwith density",
              "C3_HD_gt_LD": "C3 hyposensitive gain\nHD > LD",
              "C4_sopt_positive": "C4 s_opt > 0\n(partial response best)"}
    keys = ["C1_tragedy", "C2_monotonic", "C3_HD_gt_LD", "C4_sopt_positive"]
    vals = [mc["robust"][k] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color="#2CA02C")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([labels[k] for k in keys], fontsize=8)
    ax.set_ylim(0, 105); ax.set_ylabel("Robustness (% of interior samples)")
    ax.set_title(f"(d) Robustness over {mc['n_interior']} interior samples "
                 f"(of {mc['n_sampled']})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def _loss_sample(mc, N):
    """从 envelope 无法直接取单密度样本，这里用保存的 quant 近似；
    实际分布在 global_monte_carlo 内收集，重新生成以便绘图。"""
    # 用全局重算得到的分布（见 run 中传入）
    return mc["loss_samples"][N]


def run_sensitivity(base_dir: str = None, n: int = 6000, verbose: bool = True):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(data_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

    df_oat, df_tor = oat_scan()
    mc = global_monte_carlo(n)

    # 为绘图收集单密度损失样本（重跑一次收集，保证直方图可用）
    arr = sample_params(n)
    samples = {100: [], 400: [], 800: []}
    for i in range(n):
        p = GameParams(**{k: float(arr[k][i]) for k in arr})
        s_opt = find_population_optimum(800, p); ess800 = find_ess(800, p)
        if not ((0 < s_opt < 0.999) and (0 < ess800 < 0.999)):
            continue
        for N in samples:
            samples[N].append(yield_loss(N, p))
    mc["loss_samples"] = samples

    # 落盘
    df_oat.to_csv(os.path.join(data_dir, "sensitivity_oat.csv"),
                  index=False, encoding="utf-8-sig")
    df_tor.to_csv(os.path.join(data_dir, "sensitivity_tornado.csv"),
                  index=False, encoding="utf-8-sig")
    qrows = [{"density": N, "p5": round(q["p5"], 2), "p50": round(q["p50"], 2),
              "p95": round(q["p95"], 2), "mean": round(q["mean"], 2)}
             for N, q in mc["quant"].items()]
    pd.DataFrame(qrows).to_csv(
        os.path.join(data_dir, "sensitivity_loss_quantiles.csv"),
        index=False, encoding="utf-8-sig")
    rrows = [{"conclusion": k, "robustness_pct": round(v, 2),
              "n_interior": mc["n_interior"]}
             for k, v in mc["robust"].items()]
    pd.DataFrame(rrows).to_csv(
        os.path.join(data_dir, "sensitivity_robustness.csv"),
        index=False, encoding="utf-8-sig")

    fig_path = os.path.join(fig_dir, "Fig10_sensitivity.png")
    plot_sensitivity(df_oat, df_tor, mc, fig_path)

    if verbose:
        print("=" * 72)
        print(f"全局参数敏感性分析（Monte Carlo，{n} 组采样）")
        print("=" * 72)
        print(f"内点（非退化）样本：{mc['n_interior']}/{mc['n_sampled']}")
        print("\n定性结论稳健率（%）：")
        name_map = {"C1_tragedy": "悲剧存在 ESS>s_opt",
                    "C2_monotonic": "gap/损失随密度单调",
                    "C3_HD_gt_LD": "弱敏感增产 HD>LD",
                    "C4_sopt_positive": "s_opt>0 非零最优"}
        for k, v in mc["robust"].items():
            print(f"  {name_map[k]:28s}: {v:.1f}%")
        print("\n产量损失区间（5/50/95 分位，%）：")
        for N in [100, 400, 800]:
            q = mc["quant"][N]
            print(f"  N={N:4d}: {q['p5']:.2f} / {q['p50']:.2f} / {q['p95']:.2f}")
        print("\nOAT 龙卷风（N=800 损失极差，越大越敏感）：")
        for r in df_tor.sort_values('range', ascending=False).itertuples():
            print(f"  {r.param:6s}: range={r.range:5.2f} pp  "
                  f"(low {r.loss_at_low:.1f}% → high {r.loss_at_high:.1f}%)")
        print(f"\n图已保存: {fig_path}")
        print("=" * 72)

    return {"df_oat": df_oat, "df_tor": df_tor, "mc": mc, "fig": fig_path}


if __name__ == "__main__":
    run_sensitivity()
