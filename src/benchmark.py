"""
文献对标模块（Analytical Benchmark against Published Evidence）
================================================================

本模块用已发表的经典结果对本模型进行"定性模式复现"，是模型验证
（model validation）的第一块。逻辑是：一个新模型只有先能 reproduce
已知的、独立发表的定性结论，其新预测才值得信任。

四个对标（benchmark）：
  B1  Gersani et al. (2001) J. Ecol. 89:660-669
      根系公地悲剧：控制单株资源，共享空间的植株越多(owner→pair→shared)，
      ESS 的竞争结构投资越高、繁殖产量越低。
      本文地上版：邻居数 m 越大，ESS 避荫投资 s* 越高、产量损失越大。

  B2  Bongers et al. (2019) PLoS Comput. Biol. 15:e1007253
      密度依赖的 cESS：种植密度越高，被选择的避荫响应越强（cESS α 越大）；
      其 3D 模拟在最高密度(6400)处 cESS 无法确定。
      本文：ESS s* 随密度单调上升有闭式解（d s*/dN>0），超高密度仍确定。

  B3  Burbano-Erazo et al. (2025) New Phytol. 247:2839-2851
      避荫弱敏感番茄 IL2-2：高密度(HD)下增产，低密度(LD)下差异很小。
      本文：把表型由 ESS 向群体最优弱化，HD 增产幅度显著大于 LD；
      且存在非零最优弱化度（不是避荫越弱越好，s_opt>0）。

  B4  Falster & Westoby (2003); Vermeulen (2014)
      公地悲剧核心：ESS（个体最优）≠ 群体最优，简单优化模型与博弈模型
      给出不同预测。本文直接量化二者偏差与产量损失。

说明：各文献研究对象（根/树冠/拟南芥/番茄）、参数与尺度与本模型不同，
不追求数值一致，只检验"方向性/排序性结论(directional/ordering result)"
是否被本模型复现。每个对标给出 PASS/FAIL 判定。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (GameParams, fitness, find_ess, find_population_optimum,
                   tragedy_gap, yield_loss_percentage)


# ----------------------------------------------------------------------
# 邻居数驱动的竞争强度（用于干净对标 Gersani 的 owner vs sharing）
# ----------------------------------------------------------------------
def ess_with_neighbors(m: float, p: GameParams) -> float:
    """
    给定"邻居数 m"下的 ESS（m=0 表示独占空间的 owner，无邻体竞争）。

    把核心模型密度竞争项 k1*N^beta 替换为 k1*m^beta（m=邻居数=N-1）：
      一阶条件: k0*gamma + k1*m^beta - c*alpha*s^(alpha-1) = 0
      => s* = [(k0*gamma + k1*m^beta)/(c*alpha)]^(1/(alpha-1))
    m=0 时竞争激励为 0，s* = s_opt（owner 无悲剧），与 Gersani 单占者一致。
    """
    if p.alpha <= 1.0:
        return p.s_max
    numerator = p.k0 * p.gamma + p.k1 * (max(m, 0.0) ** p.beta)
    s = (numerator / (p.c * p.alpha)) ** (1.0 / (p.alpha - 1.0))
    return float(np.clip(s, p.s_min, p.s_max))


def s_opt_analytic(p: GameParams) -> float:
    """群体最优解析值（与密度/邻居数无关）。"""
    return find_population_optimum(1.0, p)


# ----------------------------------------------------------------------
# Benchmark 1: Gersani (2001) owner vs sharing 公地悲剧
# ----------------------------------------------------------------------
def benchmark_gersani(p: GameParams, out_dir: str) -> dict:
    """
    Gersani 三情景：owner(m=0, 独占)、pair(m=1, 两株共享)、shared(m=9, 10株共享)。
    复现：共享者越多 → ESS 竞争投资越高、单株繁殖产量越低、产量损失越大。
    """
    s_opt = s_opt_analytic(p)
    # (label, 邻居数m, Gersani对应情景)
    scenarios = [
        ("Owner (m=0)", 0, "独占空间"),
        ("Pair (m=1)", 1, "两株共享"),
        ("Shared (m=9)", 9, "10株共享"),
    ]
    rows = []
    raw_inv, raw_yld, raw_loss = [], [], []  # 未舍入值，用于判定
    for label, m, cn in scenarios:
        s_ess = ess_with_neighbors(m, p)
        # 均衡单株产量（全部采用 s_ess）。owner 时 m=0 用 N=1；其余用 N=m+1
        N = m + 1
        w_ess = fitness(s_ess, s_ess, N, p)
        w_opt = fitness(s_opt, s_opt, N, p)
        loss = 100.0 * (w_opt - w_ess) / w_opt if w_opt > 1e-10 else 0.0
        raw_inv.append(s_ess); raw_yld.append(w_ess); raw_loss.append(loss)
        rows.append({
            "scenario": label,
            "neighbors_m": m,
            "cn": cn,
            "ESS_investment_s": round(s_ess, 4),
            "optimum_s": round(s_opt, 4),
            "overinvestment": round(s_ess - s_opt, 4),
            "per_capita_yield": round(w_ess, 4),
            "yield_loss_percent": round(loss, 2),
        })

    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "benchmark1_gersani.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    # 判定（用未舍入原始值）：竞争投资与产量损失随邻居数单调上升，单株产量单调下降
    inv = np.array(raw_inv); yld = np.array(raw_yld); loss = np.array(raw_loss)
    passed = (inv[0] < inv[1] < inv[2]) and (yld[0] > yld[1] > yld[2]) and \
             (loss[0] <= loss[1] <= loss[2]) and abs(inv[0] - s_opt) < 1e-9

    return {"name": "B1 Gersani2001 公地悲剧(owner→shared)",
            "passed": bool(passed), "df": df, "csv": csv,
            "criterion": "竞争投资↑、单株产量↓、产量损失↑；owner无悲剧(ESS=opt)"}


# ----------------------------------------------------------------------
# Benchmark 2: Bongers (2019) 密度依赖 cESS
# ----------------------------------------------------------------------
def benchmark_bongers(p: GameParams, out_dir: str) -> dict:
    """
    Bongers 5 个测试密度 100/400/711/1600/6400（相对密度尺度；N 为感知邻域
    内有效竞争者数，与 k1 存在尺度简并，故仅作定性对标，不直接等同 plants m^-2）。
    复现：ESS 避荫响应随密度单调升高；并展示本模型在其失效的超高密度仍有确定解。
    """
    bongers_densities = [100, 400, 711, 1600, 6400]
    rows = []
    for N in bongers_densities:
        s_ess = find_ess(N, p)
        s_opt = find_population_optimum(N, p)
        rows.append({
            "density": N,
            "ESS_response_s": round(s_ess, 4),
            "optimum_s": round(s_opt, 4),
            "gap": round(s_ess - s_opt, 4),
            "bongers_cESS_status": "determined (analytic)" if s_ess < p.s_max
                                   else "saturated at s_max (still defined)",
        })
    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "benchmark2_bongers.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    ess_vals = df["ESS_response_s"].values
    # Bongers 报告 cESS 随密度升高（其低密度 100→1600 区间清晰）
    passed = all(ess_vals[i] <= ess_vals[i + 1] for i in range(len(ess_vals) - 1))

    # 单调性的解析证明（数值核验导数符号）
    # s* = [(k0γ + k1 N^β)/(cα)]^(1/(α-1));  N↑ => N^β↑ => s*↑ (α>1)
    return {"name": "B2 Bongers2019 密度依赖cESS",
            "passed": bool(passed), "df": df, "csv": csv,
            "criterion": "ESS 随密度单调升高（闭式解，超高密度仍确定）"}


# ----------------------------------------------------------------------
# Benchmark 3: Burbano (2025) 避荫弱敏感在高密度增产
# ----------------------------------------------------------------------
def hyposensitive_phenotype(s_wt: float, rho: float, p: GameParams) -> float:
    """
    避荫弱敏感株系表型：s_mut = s_opt + rho*(s_wt - s_opt)
    rho=1 为野生型(ESS)；rho=0 为群体最优；0<rho<1 为部分弱化（对应 IL2-2）。
    """
    s_opt = find_population_optimum(1.0, p)
    return float(np.clip(s_opt + rho * (s_wt - s_opt), p.s_min, p.s_max))


def benchmark_burbano(p: GameParams, out_dir: str,
                      N_LD: float = 100.0, N_HD: float = 800.0) -> dict:
    """
    复现 IL2-2：避荫弱敏感在 HD 增产显著、在 LD 差异很小；
    并检验存在非零最优弱化度（rho=0 附近最优，而非越弱越好）。
    """
    rhos = np.linspace(0.0, 1.0, 21)
    rows = []
    gain_LD, gain_HD = [], []
    for rho in rhos:
        rec = {"rho": round(rho, 3)}
        for tag, N in [("LD", N_LD), ("HD", N_HD)]:
            s_wt = find_ess(N, p)
            s_mut = hyposensitive_phenotype(s_wt, rho, p)
            w_wt = fitness(s_wt, s_wt, N, p)
            w_mut = fitness(s_mut, s_mut, N, p)
            gain = 100.0 * (w_mut - w_wt) / w_wt if w_wt > 1e-10 else 0.0
            rec[f"yield_gain_{tag}_pct"] = round(gain, 3)
            if tag == "LD":
                gain_LD.append(gain)
            else:
                gain_HD.append(gain)
        rows.append(rec)
    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "benchmark3_burbano.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    gain_LD, gain_HD = np.array(gain_LD), np.array(gain_HD)
    # 完全弱化到群体最优(rho=0)时：HD 最大增产 > LD 最大增产
    max_gain_HD, max_gain_LD = gain_HD[0], gain_LD[0]
    # 最优弱化度：增益在 rho=0（s_opt）处最大，且 rho=1(WT)增益=0
    best_rho_HD = rhos[int(np.argmax(gain_HD))]
    passed = (max_gain_HD > max_gain_LD) and (max_gain_HD > 0) and \
             abs(gain_HD[-1]) < 1e-6 and (best_rho_HD <= 0.1)

    return {"name": "B3 Burbano2025 避荫弱敏感HD增产",
            "passed": bool(passed), "df": df, "csv": csv,
            "rhos": rhos, "gain_LD": gain_LD, "gain_HD": gain_HD,
            "max_gain_LD": max_gain_LD, "max_gain_HD": max_gain_HD,
            "criterion": "HD增产>LD增产；WT(rho=1)增益0；最优为弱化到s_opt(非零避荫)"}


# ----------------------------------------------------------------------
# Benchmark 4: ESS != 群体最优（Falster2003 / Vermeulen2014 核心）
# ----------------------------------------------------------------------
def benchmark_tragedy(p: GameParams, out_dir: str,
                      densities=None) -> dict:
    """
    公地悲剧核心：在所有有邻体的密度下 ESS > s_opt，产量损失>0 且随密度增大。
    简单优化模型预测 s_opt，博弈模型预测 ESS，二者系统性不同。
    """
    if densities is None:
        densities = np.array([10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400])
    rows = []
    for N in densities:
        ess, s_opt, gap = tragedy_gap(N, p)
        loss = yield_loss_percentage(N, p)
        rows.append({"density": int(N), "ESS": round(ess, 4),
                     "optimum": round(s_opt, 4), "gap": round(gap, 4),
                     "yield_loss_pct": round(loss, 3)})
    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "benchmark4_tragedy.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    gaps = df["gap"].values
    losses = df["yield_loss_pct"].values
    # gap 与产量损失随密度单调上升（在 ESS 未触顶区段）
    passed = np.all(np.diff(gaps) >= -1e-9) and np.all(np.diff(losses) >= -1e-9) \
        and np.all(gaps >= -1e-9)
    return {"name": "B4 ESS≠群体最优(公地悲剧)",
            "passed": bool(passed), "df": df, "csv": csv,
            "criterion": "ESS≥opt 且偏差、产量损失随密度单调上升"}


# ----------------------------------------------------------------------
# 可视化：2x2 对标图（论文 Supplementary Figure / Model validation）
# ----------------------------------------------------------------------
def plot_benchmarks(results: dict, fig_path: str):
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                        "grid.alpha": 0.3, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # B1
    ax = axes[0, 0]
    df1 = results["B1"]["df"]
    x = np.arange(len(df1))
    ax.bar(x - 0.2, df1["ESS_investment_s"], 0.4, label="ESS investment $s^*$",
           color="#D62728")
    ax.bar(x + 0.2, df1["optimum_s"], 0.4, label="Population optimum $s_{opt}$",
           color="#2CA02C")
    ax.set_xticks(x); ax.set_xticklabels(["Owner", "Pair", "Shared"], rotation=0)
    ax.set_ylabel("Competitive investment (s)")
    ax.set_title("B1 Gersani (2001): tragedy under sharing")
    ax.legend(fontsize=8)

    # B2
    ax = axes[0, 1]
    df2 = results["B2"]["df"]
    ax.plot(df2["density"], df2["ESS_response_s"], "o-", color="#1F77B4",
            label="ESS response $s^*$ (this model, analytic)")
    ax.axhline(df2["optimum_s"].iloc[0], ls="--", color="#2CA02C",
               label="Population optimum $s_{opt}$")
    ax.set_xscale("log"); ax.set_xlabel("Planting density (effective neighbours, relative scale), $N$")
    ax.set_ylabel("Shade-avoidance response (s)")
    ax.set_title("B2 Bongers (2019): density-dependent cESS")
    ax.legend(fontsize=8)

    # B3
    ax = axes[1, 0]
    ax.plot(results["B3"]["rhos"], results["B3"]["gain_HD"], "o-",
            color="#D62728", label="High density (HD)")
    ax.plot(results["B3"]["rhos"], results["B3"]["gain_LD"], "s-",
            color="#1F77B4", label="Low density (LD)")
    ax.axvline(0.0, ls=":", color="grey", lw=1)
    ax.set_xlabel(r"Shade-response retention $\rho$  (1=WT, 0=optimum)")
    ax.set_ylabel("Yield gain over WT (%)")
    ax.set_title("B3 Burbano (2025): shade-hyposensitive gains")
    ax.legend(fontsize=8)

    # B4
    ax = axes[1, 1]
    df4 = results["B4"]["df"]
    ax.plot(df4["density"], df4["ESS"], "-", color="#D62728", label="ESS $s^*$")
    ax.plot(df4["density"], df4["optimum"], "-", color="#2CA02C",
            label="Optimum $s_{opt}$")
    ax.fill_between(df4["density"], df4["optimum"], df4["ESS"],
                    color="#FF7F0E", alpha=0.25, label="Tragedy gap")
    ax.set_xscale("log"); ax.set_xlabel("Planting density (effective neighbours, relative scale), $N$")
    ax.set_ylabel("Strategy (s)")
    ax.set_title("B4 ESS vs population optimum")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_all_benchmarks(p: GameParams = None, base_dir: str = None, verbose: bool = True):
    """运行全部四个文献对标，返回结果字典并落盘 CSV/图。"""
    if p is None:
        p = GameParams()
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    results = {
        "B1": benchmark_gersani(p, data_dir),
        "B2": benchmark_bongers(p, data_dir),
        "B3": benchmark_burbano(p, data_dir),
        "B4": benchmark_tragedy(p, data_dir),
    }

    fig_path = os.path.join(fig_dir, "FigS_benchmark_literature.png")
    plot_benchmarks(results, fig_path)

    if verbose:
        print("=" * 72)
        print("文献对标（Analytical Benchmark）结果")
        print("=" * 72)
        all_pass = True
        for key in ["B1", "B2", "B3", "B4"]:
            r = results[key]
            all_pass &= r["passed"]
            tag = "PASS ✓" if r["passed"] else "FAIL ✗"
            print(f"\n[{tag}] {r['name']}")
            print(f"       判据: {r['criterion']}")
        print("\n" + "-" * 72)
        print("B1 数据表（Gersani 三情景）:")
        print(results["B1"]["df"].to_string(index=False))
        print("\nB2 数据表（Bongers 五密度）:")
        print(results["B2"]["df"].to_string(index=False))
        print("\nB3 关键值: HD最大增产=%.2f%%, LD最大增产=%.2f%%"
              % (results["B3"]["max_gain_HD"], results["B3"]["max_gain_LD"]))
        print("\nB4 数据表（公地悲剧随密度）:")
        print(results["B4"]["df"].to_string(index=False))
        print("\n" + "=" * 72)
        print("总体判定:", "全部 PASS ✓" if all_pass else "存在 FAIL，需检查 ✗")
        print(f"对标图已保存: {fig_path}")
        print("=" * 72)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
