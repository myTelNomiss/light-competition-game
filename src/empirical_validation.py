"""
实证反演与对标模块（Empirical Inversion & Validation）
======================================================

用 Burbano-Erazo et al. (2025, New Phytol 247:2839) Fig 6 的番茄田间数据
（M82 野生型 vs 避荫弱敏感渐渗系 IL2-2；Italy/Spain；LD=3、HD=6 plants m^-2）
对理论模型做"独立实验点 × 理论预测"对标，并反演 IL2-2 的分子增益 g。

尺度无关的解析桥梁（α=2 时可闭式推出，数值用 model.fitness 复核）
----------------------------------------------------------------
均衡（s_i=s_bar=s）下 W(s)=k0(1+γs)−c s²，令
    s_opt = k0γ/(2c)，s_ess = s_opt + δ，δ = k1 N^β/(2c)，
    s_g   = s_opt + g δ（g=1 野生型，g=0 群体最优）。
则  W(s_opt) − W(s_ess) = c δ²，故
    野生型悲剧损失率   L = [W(s_opt)−W(s_ess)]/W(s_opt)
    弱敏感株增产率     G = [W(s_g)−W(s_ess)]/W(s_ess) = (1−g²)·L/(1−L)

给定田间实测增产 G_obs，可反演 g 与 L 的可行组合：
    L = G / (1 − g² + G)，  g = sqrt(1 − G(1−L)/L)
=> 完全关闭避荫(g=0)所需的最小悲剧损失 L_min = G/(1+G)。

数据为从 Fig 6 箱线图数字化的中位数（approximate, digitized），误差约
±4 t/ha、±3 cm；统计显著性以原文 Duncan 字母（P<0.05）为准，不以数字化
数值为准。Spain-LD 的 M82 为低产异常点（字母 b，n=4），主反演以唯一统计
显著的 HD-Spain 点（IL2-2 为 A、M82 为 B）为锚，其余点作不确定性带。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GameParams, fitness, find_ess, find_population_optimum


DATA_NOTE = ("Digitized medians from Burbano-Erazo et al. 2025 New Phytol "
             "247:2839 Fig6; approximate, see CSV notes. Significance from "
             "original Duncan letters (P<0.05).")


def load_digitized(base_dir: str):
    data_dir = os.path.join(base_dir, "data")
    dy = pd.read_csv(os.path.join(data_dir, "burbano2025_fig6_yield_digitized.csv"))
    dh = pd.read_csv(os.path.join(data_dir, "burbano2025_fig6_height_digitized.csv"))
    return dy, dh


def location_specific_ratios(df, value_col):
    """计算每地点×密度下 IL2-2 相对 M82 的百分比差异。"""
    rows = []
    for (density, loc), g in df.groupby(["density", "location"]):
        wt = g.loc[g.genotype == "M82", value_col].values[0]
        mut = g.loc[g.genotype == "IL2-2", value_col].values[0]
        n_wt = g.loc[g.genotype == "M82", "n"].values[0]
        n_mut = g.loc[g.genotype == "IL2-2", "n"].values[0]
        rows.append({"density": density, "location": loc,
                     "M82": wt, "IL2-2": mut,
                     "pct_diff": 100.0 * (mut - wt) / wt,
                     "n_M82": n_wt, "n_IL22": n_mut})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 解析桥梁
# ----------------------------------------------------------------------
def gain_from_loss(L, g):
    """弱敏感株增产率 G = (1-g^2)*L/(1-L)。L、G 为小数。"""
    return (1.0 - g ** 2) * L / (1.0 - L)


def loss_from_gain(G, g):
    """由实测增产 G 与增益 g 反演野生型悲剧损失 L。"""
    return G / (1.0 - g ** 2 + G)


def gain_from_gain(G, L):
    """由实测增产 G 与悲剧损失 L 反演增益 g。"""
    val = 1.0 - G * (1.0 - L) / L
    return float(np.sqrt(max(val, 0.0)))


def model_gain_vs_loss(gp: GameParams, g: float, densities=None):
    """数值（用 model.fitness）扫描密度，返回 (L_WT, G_g)，消去密度轴。"""
    if densities is None:
        densities = np.logspace(np.log10(20), np.log10(4000), 200)
    Ls, Gs = [], []
    for N in densities:
        s_opt = find_population_optimum(N, gp)
        s_ess = find_ess(N, gp)
        s_g = s_opt + g * (s_ess - s_opt)
        w_opt = fitness(s_opt, s_opt, N, gp)
        w_ess = fitness(s_ess, s_ess, N, gp)
        w_g = fitness(s_g, s_g, N, gp)
        L = (w_opt - w_ess) / w_opt
        G = (w_g - w_ess) / w_ess
        Ls.append(L); Gs.append(G)
    return np.array(Ls), np.array(Gs)


# ----------------------------------------------------------------------
# 主分析
# ----------------------------------------------------------------------
def run_inversion(base_dir: str = None, gp: GameParams = None, verbose=True):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gp is None:
        gp = GameParams()
    data_dir = os.path.join(base_dir, "data")

    dy, dh = load_digitized(base_dir)
    ry = location_specific_ratios(dy, "yield_t_ha_median")
    rh = location_specific_ratios(dh, "plant_height_cm_median")

    # 唯一统计显著的增产点：HD-Spain（IL2-2=A, M82=B）
    G_hd_spain = ry.loc[(ry.density == "HD") & (ry.location == "Spain"),
                        "pct_diff"].values[0] / 100.0
    G_hd_italy = ry.loc[(ry.density == "HD") & (ry.location == "Italy"),
                        "pct_diff"].values[0] / 100.0
    G_ld_italy = ry.loc[(ry.density == "LD") & (ry.location == "Italy"),
                        "pct_diff"].values[0] / 100.0
    G_ld_spain = ry.loc[(ry.density == "LD") & (ry.location == "Spain"),
                        "pct_diff"].values[0] / 100.0

    # 反演：不同分子先验 g 下，番茄 HD 的野生型悲剧损失 L
    g_priors = {"g=0 (full suppression)": 0.0,
                "g=0.35 (pif457-like)": 0.35,
                "g=0.55 (IL2-2 auxin)": 0.55,
                "g=0.70 (mild PIF)": 0.70}
    inv_rows = []
    for label, g in g_priors.items():
        L = loss_from_gain(G_hd_spain, g)
        inv_rows.append({"assumed_gain_g": g, "scenario": label,
                         "inferred_WT_tragedy_loss_pct": round(100 * L, 1),
                         "observed_HD_Spain_gain_pct": round(100 * G_hd_spain, 1)})
    df_inv = pd.DataFrame(inv_rows)
    L_min = G_hd_spain / (1 + G_hd_spain)  # g=0 时的最小 L

    # 落盘
    ry_out = ry.copy(); ry_out["pct_diff"] = ry_out["pct_diff"].round(1)
    rh_out = rh.copy(); rh_out["pct_diff"] = rh_out["pct_diff"].round(1)
    ry_out.to_csv(os.path.join(data_dir, "burbano_ratios_yield.csv"),
                  index=False, encoding="utf-8-sig")
    rh_out.to_csv(os.path.join(data_dir, "burbano_ratios_height.csv"),
                  index=False, encoding="utf-8-sig")
    df_inv.to_csv(os.path.join(data_dir, "burbano_inversion_g_loss.csv"),
                  index=False, encoding="utf-8-sig")

    if verbose:
        print("=" * 72)
        print("Burbano (2025) 田间数据反演")
        print("=" * 72)
        print("\nIL2-2 相对 M82 的产量差异（%，地点内）:")
        print(ry_out.to_string(index=False))
        print("\nIL2-2 相对 M82 的株高差异（%，地点内）:")
        print(rh_out.to_string(index=False))
        print(f"\n锚点（唯一统计显著）：HD-Spain 增产 = {100*G_hd_spain:.1f}%")
        print(f"HD-Italy = {100*G_hd_italy:.1f}%（不显著）；"
              f"LD-Italy = {100*G_ld_italy:.1f}%（不显著）；"
              f"LD-Spain = {100*G_ld_spain:.1f}%（M82 低基数异常，不纳入反演）")
        print("\n反演：番茄 HD 野生型光竞争悲剧损失 L（随分子增益假设 g）:")
        print(df_inv.to_string(index=False))
        print(f"\n=> 完全关闭避荫(g=0)所需最小 L = {100*L_min:.1f}%；")
        print(f"   分子先验 g∈[0.35,0.55] 对应 L∈"
              f"[{100*loss_from_gain(G_hd_spain,0.55):.1f}%,"
              f"{100*loss_from_gain(G_hd_spain,0.35):.1f}%]")
        print("=" * 72)

    return {"ry": ry, "rh": rh, "df_inv": df_inv,
            "G_hd_spain": G_hd_spain, "G_hd_italy": G_hd_italy,
            "G_ld_italy": G_ld_italy, "G_ld_spain": G_ld_spain,
            "L_min": L_min, "g_priors": g_priors}


# ----------------------------------------------------------------------
# 可视化：2x2 实证对标图
# ----------------------------------------------------------------------
def plot_validation(res: dict, gp: GameParams, fig_path: str):
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                        "grid.alpha": 0.3, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    dy, dh = res["ry"], res["rh"]

    # (a) 产量复刻（地点内分组）
    ax = axes[0, 0]
    groups = [("LD", "Italy"), ("LD", "Spain"), ("HD", "Italy"), ("HD", "Spain")]
    labels = ["LD\nItaly", "LD\nSpain", "HD\nItaly", "HD\nSpain"]
    x = np.arange(len(groups)); w = 0.38
    m82 = [dy.loc[(dy.density == d) & (dy.location == l), "M82"].values[0]
           for d, l in groups]
    il2 = [dy.loc[(dy.density == d) & (dy.location == l), "IL2-2"].values[0]
           for d, l in groups]
    ax.bar(x - w/2, m82, w, label="M82 (WT)", color="#7F7F7F")
    ax.bar(x + w/2, il2, w, label="IL2-2 (shade-hyposensitive)", color="#D62728")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Total ripe fruit yield (t ha$^{-1}$)")
    ax.set_ylim(0, 188)
    ax.set_title("(a) Field yield (Burbano 2025, Fig6b digitized)")
    # 在 HD-Spain 的 IL2-2 柱顶直接标注唯一显著差异
    ax.text(3 + w/2, il2[3] + 3, "+13.5%\n(A vs B)", ha="center", va="bottom",
            fontsize=8.5, color="#D62728", fontweight="bold")
    ax.text(1 + w/2, il2[1] + 3, "M82 low\noutlier", ha="center", va="bottom",
            fontsize=7, color="#7F7F7F")
    ax.legend(fontsize=8, loc="upper left")

    # (b) 株高（表型佐证）
    ax = axes[0, 1]
    m82h = [dh.loc[(dh.density == d) & (dh.location == l), "M82"].values[0]
            for d, l in groups]
    il2h = [dh.loc[(dh.density == d) & (dh.location == l), "IL2-2"].values[0]
            for d, l in groups]
    ax.bar(x - w/2, m82h, w, label="M82 (WT)", color="#7F7F7F")
    ax.bar(x + w/2, il2h, w, label="IL2-2", color="#FF7F0E")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Plant height (cm)")
    ax.set_title("(b) Height: IL2-2 elongates less under HD")
    ax.legend(fontsize=8)

    # (c) 尺度无关理论对标：G vs L 曲线族 + HD-Spain 实测约束
    ax = axes[1, 0]
    L_grid = np.linspace(0.001, 0.25, 200)
    colors = {0.0: "#2CA02C", 0.35: "#9467BD", 0.55: "#D62728", 0.70: "#8C564B"}
    glab = {0.0: "g=0 (optimum)", 0.35: "g=0.35 (pif457)",
            0.55: "g=0.55 (IL2-2)", 0.70: "g=0.70 (mild)"}
    for g in [0.0, 0.35, 0.55, 0.70]:
        ax.plot(100 * L_grid, 100 * gain_from_loss(L_grid, g),
                color=colors[g], label=glab[g], lw=2 if g == 0.55 else 1.3)
    Gobs = 100 * res["G_hd_spain"]
    ax.axhline(Gobs, ls="--", color="#D62728", lw=1.4)
    ax.text(1, Gobs + 0.4, f"HD-Spain observed = {Gobs:.1f}%",
            color="#D62728", fontsize=8)
    # v5: 区分两个层次——g=0 给出不依赖分子先验的数据驱动硬下界；
    # g=0.35–0.55（分子表征先验）给出条件区间。二者不得混为一个带。
    L_hard = 100 * res["L_min"]                       # g=0: data-only lower bound (11.9%)
    L_cond_lo = 100 * loss_from_gain(res["G_hd_spain"], 0.35)  # pif457
    L_cond_hi = 100 * loss_from_gain(res["G_hd_spain"], 0.55)  # IL2-2
    # 条件区间（molecular priors）
    ax.axvspan(L_cond_lo, L_cond_hi, color="#D62728", alpha=0.15)
    ax.text((L_cond_lo + L_cond_hi) / 2, 2.5,
            f"conditional L\n{L_cond_lo:.0f}–{L_cond_hi:.0f}% (g=0.35–0.55)",
            ha="center", fontsize=7.5, color="#D62728")
    # 数据驱动硬下界（g=0）
    ax.axvline(L_hard, color="#444444", ls=":", lw=1.6)
    ax.text(L_hard, 20.6, f"data-only lower bound\nL ≥ {L_hard:.1f}% (g=0)",
            ha="center", va="top", fontsize=7.5, color="#444444")
    # LD 点（增益不显著，置于 L 很小处的 0 带）
    ax.axhspan(-12, 0, color="#1F77B4", alpha=0.06)
    ax.text(20, -8, "LD: gains ≈ 0 (n.s.)", fontsize=8, color="#1F77B4")
    ax.set_xlabel("Wild-type tragedy yield loss L (%)")
    ax.set_ylabel("Shade-hyposensitive yield gain G (%)")
    ax.set_ylim(-13, 22)
    ax.set_title("(c) Scale-free inversion: G = (1−g²)·L/(1−L)")
    ax.legend(fontsize=7, loc="upper left")

    # (d) 反演曲线 g(L) + 模型主粮参照
    ax = axes[1, 1]
    Ls = np.linspace(res["L_min"] + 0.002, 0.30, 200)
    gs = [gain_from_gain(res["G_hd_spain"], L) for L in Ls]
    ax.plot(100 * Ls, gs, color="#D62728", lw=2,
            label="inversion from HD-Spain G=13.5%")
    for g, c, lab in [(0.35, "#9467BD", "pif457 g=.35"),
                      (0.55, "#D62728", "IL2-2 g=.55")]:
        L = loss_from_gain(res["G_hd_spain"], g)
        ax.plot(100 * L, g, "o", color=c, ms=8)
        ax.annotate(f"{lab}\nL={100*L:.0f}%", (100 * L, g),
                    textcoords="offset points", xytext=(8, -2), fontsize=8)
    # 模型默认主粮参数参照点（N=800，L≈9.3%，g=.55 增益≈7%）
    N = 800
    s_opt = find_population_optimum(N, gp); s_ess = find_ess(N, gp)
    w_opt = fitness(s_opt, s_opt, N, gp); w_ess = fitness(s_ess, s_ess, N, gp)
    L_model = 100 * (w_opt - w_ess) / w_opt
    ax.axvline(L_model, ls=":", color="#1F77B4")
    ax.text(L_model + 0.3, 0.95, f"model staple N=800\nL={L_model:.1f}%",
            fontsize=8, color="#1F77B4")
    ax.set_xlabel("Wild-type tragedy loss L (%)")
    ax.set_ylabel("Required molecular gain g")
    ax.set_title("(d) Inferred g–L combinations for tomato HD")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_empirical_validation(base_dir: str = None, gp: GameParams = None,
                             verbose: bool = True):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gp is None:
        gp = GameParams()
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    res = run_inversion(base_dir, gp, verbose=verbose)
    fig_path = os.path.join(fig_dir, "Fig9_empirical_burbano.png")
    plot_validation(res, gp, fig_path)
    if verbose:
        print(f"实证对标图已保存: {fig_path}")
    return res, fig_path


if __name__ == "__main__":
    run_empirical_validation()
