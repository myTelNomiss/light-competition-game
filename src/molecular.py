"""
分子-表型-群体跨层耦合模块（Molecular-to-Canopy Coupling）
=========================================================

把光信号分子通路（Huber et al. 2021, Plant Cell Environ 44:1014, Fig 2；
Burbano-Erazo et al. 2025, New Phytol 247:2839）参数化到演化博弈的
表型策略 s，实现"光受体 → PIF → 生长素 → 避荫表型 → 群体产量"的贯通。

通路（遮荫/邻近信号 SAS）：
    低 R:FR  ──▶ phyB 由活性 Pfr 转为无活性 Pr ──▶ 解除对 PIFs 的抑制
    低蓝光   ──▶ CRY1/2 去活化 ─────────────────▶ 协同解除抑制
                         │
                         ▼
        PIF4/5/7（结合 G-box）◀─ 负调控：HY5/HFR1/PAR1/2、DELLA
                         │  诱导
                         ▼
   生长素 TAA1/YUC→IAA（+GA、BR、乙烯）──▶ 伸长、hyponasty、分枝减少、
                                            提早开花（=避荫策略 s）

跨层锚定（关键设计）
--------------------
避荫响应的"绝对幅度随密度的形状"由演化博弈 ESS 决定（自然选择的结果）；
分子通路是实现该响应的机制，其"总增益 g"决定一个基因型把野生型的避荫
可塑性响应缩放多少：

    s(N; g) = s_opt(N) + g · [ s_ESS(N) − s_opt(N) ]

  g = 1 野生型 → 精确实现 ESS（跨层一致，误差恒为 0）；
  g = 0 完全丧失避荫可塑性 → 落到群体最优 s_opt；
  0 < g < 1 避荫弱敏感（IL2-2 / pif457 / sav3 / 温和 PIF 敲低）。

总增益 g 由两段机制相乘：
  g = downstream × sensing
  downstream = PIF表达 × 生长素合成/信号能力(TAA1/YUC) × GA去抑制 × (1−负调控)
  sensing    = 该基因型光受体(phyB/CRY)对冠层遮荫信号的平均转导效率（相对WT）
正常光下 shade_signal≈0，故弱化 g 几乎不改变无遮荫表型——这正是"弱敏感
(hypo-sensitive)"而非"组成型矮化"，对应 Huber(2021) 的"微调而非整体关闭"。

分子参数为可校准相对量；默认值为示例标定，绝对值需用公共数据与文献区间
（AraPheno / 1001 Genomes / Rice SNP-Seek；Burbano sav3-5、pif457）估计。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

from model import (GameParams, fitness, find_ess, find_population_optimum,
                   tragedy_gap)


# ----------------------------------------------------------------------
# 参数容器
# ----------------------------------------------------------------------
@dataclass
class MolecularParams:
    # —— 光环境 ——
    RFR_open: float = 1.15      # 开阔地 R:FR（冠层上方）
    blue_open: float = 1.0      # 开阔地相对蓝光（归一化=1）
    kappa_rfr: float = 0.15     # R:FR 衰减系数（LAI 亚线性增长）
    kappa_blue: float = 0.22    # 蓝光衰减系数
    lai_lambda: float = 0.60
    N_ref: float = 100.0
    # —— 光受体光平衡（Hill 函数）——
    K_phyB: float = 0.45
    h_phyB: float = 1.0
    K_cry: float = 0.40
    h_cry: float = 1.0
    # —— 信号整合权重 ——
    w_phyB: float = 0.62
    w_cry: float = 0.26
    w_basal: float = 0.12
    # —— 下游分子增益组分（野生型=1，突变体按倍数改）——
    PIF_expression: float = 1.0     # PIF4/5/7 有效表达/活性
    auxin_capacity: float = 1.0     # TAA1/YUC→IAA 合成与信号能力
    GA_factor: float = 1.0          # GA 对伸长的去抑制（拮抗 DELLA）
    negative_reg: float = 0.0       # HY5/HFR1/PAR 负调控强度 0..1


# 用于"信号感知效率"对比的野生型参考参数（默认光受体、下游全 1）
def wt_reference() -> MolecularParams:
    return MolecularParams()


# ----------------------------------------------------------------------
# 光环境与光受体
# ----------------------------------------------------------------------
def density_to_light(N: float, mp: MolecularParams):
    """密度 -> (冠层内 R:FR, 相对蓝光)，Beer-Lambert/LAI 近似。"""
    lai = (max(N, 1.0) / mp.N_ref) ** mp.lai_lambda
    rfr = mp.RFR_open / (1.0 + mp.kappa_rfr * lai)
    blue = mp.blue_open / (1.0 + mp.kappa_blue * lai)
    return rfr, blue


def phyB_pfr_fraction(rfr: float, mp: MolecularParams) -> float:
    """phyB 活性 Pfr 比例（Hill 光平衡）：R:FR 越高，Pfr 越多，越抑制避荫。"""
    x = rfr ** mp.h_phyB
    return x / (mp.K_phyB ** mp.h_phyB + x)


def cry_activity(blue: float, mp: MolecularParams) -> float:
    """CRY 活性（随相对蓝光饱和）。"""
    x = blue ** mp.h_cry
    return x / (mp.K_cry ** mp.h_cry + x)


def shade_signal(N: float, mp: MolecularParams) -> float:
    """冠层遮荫/邻近信号强度 ∈[0,1]：低 R:FR、低蓝光经 phyB、CRY 去抑制 PIF。"""
    rfr, blue = density_to_light(N, mp)
    pfr = phyB_pfr_fraction(rfr, mp)
    cry = cry_activity(blue, mp)
    sig = (mp.w_phyB * (1.0 - pfr) +
           mp.w_cry * (1.0 - cry) +
           mp.w_basal)
    return float(np.clip(sig, 0.0, 1.0))


# ----------------------------------------------------------------------
# 分子增益 g = downstream × sensing
# ----------------------------------------------------------------------
def downstream_factor(mp: MolecularParams) -> float:
    """下游转导能力：PIF × 生长素 × GA × 负调控。"""
    return float(mp.PIF_expression * mp.auxin_capacity * mp.GA_factor *
                 (1.0 - np.clip(mp.negative_reg, 0.0, 0.95)))


def sensing_factor(mp: MolecularParams, ref: MolecularParams = None,
                   densities=None) -> float:
    """
    光受体信号感知效率（相对野生型参考）：在密度梯度上平均
    signal_mut(N)/signal_WT(N)。phyB 过表达等信号端改变由此体现。
    """
    if ref is None:
        ref = wt_reference()
    if densities is None:
        densities = np.array([100, 200, 400, 800, 1600])
    sig_mut = np.array([shade_signal(N, mp) for N in densities])
    sig_wt = np.array([shade_signal(N, ref) for N in densities])
    denom = np.sum(sig_wt ** 2)
    return float(np.sum(sig_mut * sig_wt) / denom) if denom > 0 else 1.0


def effective_gain(mp: MolecularParams, ref: MolecularParams = None) -> float:
    """基因型总增益 g（WT=1）。"""
    if ref is None:
        ref = wt_reference()
    ds = downstream_factor(mp) / downstream_factor(ref)
    se = sensing_factor(mp, ref)
    return float(np.clip(ds * se, 0.0, 1.5))


# ----------------------------------------------------------------------
# 分子 -> 表型 s（以 WT-ESS 为锚）
# ----------------------------------------------------------------------
def phenotype_strategy(N: float, mp: MolecularParams,
                       gp: GameParams, ref: MolecularParams = None) -> float:
    """s(N;g) = s_opt + g·(ESS − s_opt)，g 为该基因型分子总增益。"""
    g = effective_gain(mp, ref)
    s_opt = find_population_optimum(N, gp)
    ess = find_ess(N, gp)
    s = s_opt + g * (ess - s_opt)
    return float(np.clip(s, gp.s_min, gp.s_max))


# ----------------------------------------------------------------------
# 基因型预设（对应文献突变体 / 育种材料）
# ----------------------------------------------------------------------
def genotype_presets() -> Dict[str, MolecularParams]:
    """
    WT       野生型（g=1，实现 ESS）
    IL2-2    避荫弱敏感番茄（auxin 稳态改变，TAA1/YUC/IAA/SAUR 弱化）
    pif457   拟南芥 PIF4/5/7 缺陷（Burbano Fig5：HD 避荫伸长减弱）
    sav3     拟南芥 TAA1 缺陷（sav3-5，Trp→IPA 受阻）
    weak_PIF PIF 温和部分敲低（育种中间态）
    phyB_OE  phyB 过表达（信号端策略，对比用）
    """
    return {
        "WT":       MolecularParams(),
        "IL2-2":    MolecularParams(auxin_capacity=0.55),
        "pif457":   MolecularParams(PIF_expression=0.35),
        "sav3":     MolecularParams(auxin_capacity=0.30),
        "weak_PIF": MolecularParams(PIF_expression=0.70),
        "phyB_OE":  MolecularParams(K_phyB=0.30),
    }


def genotype_yield(name: str, mp: MolecularParams, N: float,
                   gp: GameParams, ref: MolecularParams) -> dict:
    """某基因型在密度 N 的均衡表型与相对 WT-ESS 的产量增益。"""
    s = phenotype_strategy(N, mp, gp, ref)
    s_wt = find_ess(N, gp)
    w = fitness(s, s, N, gp)
    w_wt = fitness(s_wt, s_wt, N, gp)
    return {"genotype": name, "g": round(effective_gain(mp, ref), 3),
            "s": s, "yield_gain_vs_WT_pct":
            100.0 * (w - w_wt) / w_wt if w_wt > 1e-10 else 0.0}


# ----------------------------------------------------------------------
# 跨层分析
# ----------------------------------------------------------------------
def run_genotype_comparison(gp: GameParams, densities=None,
                            out_dir: str = None) -> pd.DataFrame:
    """各基因型在密度梯度上的增益 g、表型与产量（复现 Burbano 突变体梯度）。"""
    if densities is None:
        densities = np.array([100, 200, 400, 800, 1600])
    ref = wt_reference()
    genos = genotype_presets()
    rows = []
    for N in densities:
        rfr, blue = density_to_light(N, ref)
        for name, gmp in genos.items():
            r = genotype_yield(name, gmp, N, gp, ref)
            rows.append({"density": int(N), "genotype": name,
                         "RFR": round(rfr, 3), "gain_g": r["g"],
                         "phenotype_s": round(r["s"], 4),
                         "yield_gain_vs_WT_pct": round(r["yield_gain_vs_WT_pct"], 2)})
    df = pd.DataFrame(rows)
    if out_dir:
        df.to_csv(os.path.join(out_dir, "molecular_genotype_comparison.csv"),
                  index=False, encoding="utf-8-sig")
    return df


def yield_gain_at_gain(g: float, N: float, gp: GameParams) -> float:
    """把避荫增益弱化到 g（WT=1）时，相对 WT-ESS 的增产%。"""
    s_opt = find_population_optimum(N, gp)
    ess = find_ess(N, gp)
    s = s_opt + g * (ess - s_opt)
    w, w_wt = fitness(s, s, N, gp), fitness(ess, ess, N, gp)
    return 100.0 * (w - w_wt) / w_wt if w_wt > 1e-10 else 0.0


def run_breeding_targets(gp: GameParams, out_dir: str = None) -> pd.DataFrame:
    """
    分密度的分子育种靶点：报告若干增益水平（含 IL2-2 级、PIF 级、群体最优）
    的预期增产，以及达到群体最优所需的下游弱化比例。
    """
    levels = [("WT_g=1.0", 1.0), ("weak_PIF_g=0.70", 0.70),
              ("IL2-2_g=0.55", 0.55), ("pif457_g=0.35", 0.35),
              ("optimum_g=0.0", 0.0)]
    rows = []
    for N in [250, 400, 500, 800]:
        ess, s_opt, gap = tragedy_gap(N, gp)
        rec = {"density": int(N), "WT_ESS": round(ess, 3),
               "s_opt": round(s_opt, 3)}
        for label, g in levels:
            rec[label + "_gain%"] = round(yield_gain_at_gain(g, N, gp), 2)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if out_dir:
        df.to_csv(os.path.join(out_dir, "molecular_breeding_targets.csv"),
                  index=False, encoding="utf-8-sig")
    return df


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def plot_molecular(gp: GameParams, fig_path: str, densities=None):
    if densities is None:
        densities = np.logspace(np.log10(50), np.log10(3000), 80)
    ref = wt_reference()
    genos = genotype_presets()

    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                        "grid.alpha": 0.3, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) 光环境随密度
    ax = axes[0, 0]
    rfr = [density_to_light(N, ref)[0] for N in densities]
    blue = [density_to_light(N, ref)[1] for N in densities]
    ax.plot(densities, rfr, color="#D62728", label="R:FR (phyB cue)")
    ax.plot(densities, blue, color="#1F77B4", label="Relative blue (CRY cue)")
    ax.set_xscale("log"); ax.set_xlabel("Planting density (effective neighbours), $N$")
    ax.set_ylabel("Light signal"); ax.set_title("(a) Canopy light cues vs density")
    ax.legend(fontsize=8)

    # (b) 各基因型表型 s(N)，WT 应精确压在 ESS
    ax = axes[0, 1]
    colors = {"WT": "black", "IL2-2": "#D62728", "pif457": "#FF7F0E",
              "sav3": "#9467BD", "weak_PIF": "#8C564B", "phyB_OE": "#17BECF"}
    styles = {"WT": "-", "IL2-2": "-", "pif457": "--", "sav3": "--",
              "weak_PIF": ":", "phyB_OE": "-."}
    ess_curve = [find_ess(N, gp) for N in densities]
    ax.plot(densities, ess_curve, color="black", lw=2.5, alpha=0.35,
            label="ESS (evolutionary anchor)")
    for name, gmp in genos.items():
        s = [phenotype_strategy(N, gmp, gp, ref) for N in densities]
        ax.plot(densities, s, styles[name], color=colors[name],
                label=f"{name} (g={effective_gain(gmp, ref):.2f})",
                lw=2.0 if name in ("WT", "IL2-2") else 1.3)
    ax.axhline(find_population_optimum(100, gp), color="#2CA02C", ls=":",
               lw=1.5, label="$s_{opt}$ (population optimum)")
    ax.set_xscale("log"); ax.set_xlabel("Planting density (effective neighbours), $N$")
    ax.set_ylabel("Shade-avoidance phenotype s")
    ax.set_title("(b) Molecularly defined phenotypes")
    ax.legend(fontsize=7, ncol=2)

    # (c) HD vs LD 各基因型增产（复现 Burbano）
    ax = axes[1, 0]
    key_genos = ["WT", "weak_PIF", "IL2-2", "pif457", "sav3", "phyB_OE"]
    N_LD, N_HD = 100, 800
    gain_LD = [genotype_yield(n, genos[n], N_LD, gp, ref)["yield_gain_vs_WT_pct"]
               for n in key_genos]
    gain_HD = [genotype_yield(n, genos[n], N_HD, gp, ref)["yield_gain_vs_WT_pct"]
               for n in key_genos]
    x = np.arange(len(key_genos)); w = 0.38
    ax.bar(x - w/2, gain_LD, w, label="LD (N=100)", color="#1F77B4")
    ax.bar(x + w/2, gain_HD, w, label="HD (N=800)", color="#D62728")
    ax.set_xticks(x); ax.set_xticklabels(key_genos, rotation=20)
    ax.set_ylabel("Yield gain over WT (%)")
    ax.set_title("(c) Shade-hyposensitive genotypes: HD ≫ LD gains")
    ax.legend(fontsize=8)

    # (d) 增产 vs 下游弱化程度（dose-response），多密度
    ax = axes[1, 1]
    knockdown = np.linspace(0, 1, 51)  # 0=WT, 1=完全关闭
    g_grid = 1.0 - knockdown
    for N, c in [(100, "#1F77B4"), (400, "#FF7F0E"), (800, "#D62728"),
                 (1600, "#7F0000")]:
        yg = [yield_gain_at_gain(g, N, gp) for g in g_grid]
        ax.plot(100 * knockdown, yg, color=c, label=f"N={N}")
    ax.set_xlabel("Downstream (PIF/auxin) knockdown (%)")
    ax.set_ylabel("Yield gain over WT (%)")
    ax.set_title("(d) Breeding dose–response: returns rise with density")
    ax.legend(fontsize=8, title="Density")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_molecular_analysis(gp: GameParams = None, base_dir: str = None,
                           verbose: bool = True):
    if gp is None:
        gp = GameParams()
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(data_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

    ref = wt_reference()
    genos = genotype_presets()

    # 跨层一致性：WT(g=1) 分子表型必须精确等于 ESS
    check_dens = [50, 100, 400, 800, 1600]
    consist = []
    for N in check_dens:
        s_mol = phenotype_strategy(N, ref, gp, ref)
        s_ess = find_ess(N, gp)
        consist.append((N, s_mol, s_ess, abs(s_mol - s_ess)))
    max_err = max(c[3] for c in consist)

    df_geno = run_genotype_comparison(gp, out_dir=data_dir)
    df_tgt = run_breeding_targets(gp, out_dir=data_dir)

    fig_path = os.path.join(fig_dir, "Fig8_molecular_coupling.png")
    plot_molecular(gp, fig_path)

    if verbose:
        print("=" * 72)
        print("分子-表型-群体 跨层耦合结果")
        print("=" * 72)
        print("各基因型分子总增益 g（WT=1）:")
        for name, gmp in genos.items():
            print(f"  {name:8s}: g={effective_gain(gmp, ref):.3f}")
        print("\n跨层一致性（WT 分子表型 vs 博弈ESS）:")
        for N, sm, se, err in consist:
            print(f"  N={N:4d}: s_molecular={sm:.4f}, ESS={se:.4f}, 误差={err:.2e}")
        print("\n各基因型在 N=800(HD) 的表型与增产:")
        print(df_geno[df_geno.density == 800].to_string(index=False))
        print("\n分子育种靶点（不同增益水平的预期增产 %）:")
        print(df_tgt.to_string(index=False))
        print("\n跨层一致性判定:", "PASS ✓（WT 精确实现 ESS，误差<1e-9）"
              if max_err < 1e-9 else f"FAIL ✗（最大误差{max_err:.3e}）")
        print(f"图已保存: {fig_path}")
        print("=" * 72)

    return {"consistency": consist, "max_error": max_err,
            "df_geno": df_geno, "df_target": df_tgt, "fig": fig_path}


if __name__ == "__main__":
    run_molecular_analysis()
