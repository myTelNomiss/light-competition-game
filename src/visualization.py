"""
Visualization module: publication-quality figures (English)

Figure list (main text Fig. 1-7):
- Fig. 1: conceptual model framework (molecular signal -> individual strategy -> stand game -> breeding)
- Fig. 2: ESS vs population optimum across density (core result)
- Fig. 3: evolutionary convergence trajectories (individual-based validation)
- Fig. 4: pairwise invasibility plot (PIP)
- Fig. 5: frequency-dependent competition among mixed strategies
- Fig. 6: breeding strategy optimization (targets and expected yield gain)
- Fig. 7: coupling of phyB light signalling to phenotype and yield
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from typing import List, Optional, Tuple

from model import (GameParams, fitness, find_ess, find_population_optimum,
                    tragedy_gap, yield_loss_percentage, pairwise_invasibility,
                    breeding_target, light_signaling_coupling)
from simulation import EvolutionSimulation, mixed_strategy_competition


# ============================================================
# Global style
# ============================================================
def setup_style():
    """Publication-quality style."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


COLORS = {
    'ess': '#D62728',
    'opt': '#2CA02C',
    'gap': '#FF7F0E',
    'density': '#1F77B4',
    'mutant': '#9467BD',
    'resident': '#8C564B',
    'light': '#E377C2',
}

DEN_LABEL = 'Planting density (effective neighbours), $N$'
STRAT_LABEL = 'Shade-avoidance strategy, $s$'


# ============================================================
# Fig. 1: conceptual model framework
# ============================================================
def plot_model_framework(p: GameParams, save_path: str):
    """Molecular light signal -> individual strategy -> stand game -> breeding."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Four level boxes
    layers = [
        (0.5, 5.5, 2.5, 1.8, 'Molecular level\nlight signalling', '#E8F4FD'),
        (3.5, 5.5, 2.5, 1.8, 'Individual level\nshade-avoidance\nstrategy $s$', '#FFF3E0'),
        (6.5, 5.5, 2.5, 1.8, 'Stand level\nlight-competition\ngame', '#F3E5F5'),
        (9.5, 5.5, 2.0, 1.8, 'Breeding\napplication\ndense-stand yield', '#E8F5E9'),
    ]
    for x, y, w, h, label, color in layers:
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='#333',
                          linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=3)

    # Connecting arrows
    for x1, x2 in [(3.0, 3.5), (6.0, 6.5), (9.0, 9.5)]:
        ax.annotate('', xy=(x2, 6.4), xytext=(x1, 6.4),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Per-level details
    ax.text(1.75, 5.05, 'phyB (R/FR)\nCRY (blue)\nPIFs (integration)',
            ha='center', va='top', fontsize=9, color='#1565C0')
    ax.text(4.75, 5.05, '$s \\in [0,1]$\n0 = shade-tolerant\n1 = strong avoidance\nheight/petiole elongation',
            ha='center', va='top', fontsize=9, color='#E65100')
    ax.text(7.75, 5.05, 'frequency-dependent\nfitness\nESS vs population optimum\ntragedy of the commons',
            ha='center', va='top', fontsize=9, color='#6A1B9A')
    ax.text(10.5, 5.05, 'attenuate avoidance\nraise stand yield\nmolecular breeding target',
            ha='center', va='top', fontsize=9, color='#2E7D32')

    # Fitness function (correct form: k1 (s_i - sbar) N^beta)
    ax.text(6, 3.55,
            r'Individual fitness:  $W(s_i,\bar{s},N)=k_0(1+\gamma s_i)+k_1(s_i-\bar{s})\,N^{\beta}-c\,s_i^{\alpha}$',
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#F9A825'))

    # ESS vs population optimum
    ax.text(3, 1.7, 'Evolutionarily stable\nstrategy (ESS)\nendpoint of natural selection\nindividual optimum',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCDD2', edgecolor=COLORS['ess']))
    ax.text(9, 1.7, 'Population optimum\n($s_{opt}$)\nmaximum stand yield\ncollective optimum',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor=COLORS['opt']))

    ax.annotate('', xy=(7.5, 1.7), xytext=(4.5, 1.7),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gap'], lw=2.5))
    ax.text(6, 2.35, r'Tragedy gap:  $\Delta s = s^{*} - s_{opt} > 0 \; \Rightarrow$ over-investment in avoidance $\Rightarrow$ dense-stand yield loss',
            ha='center', va='bottom', fontsize=10, color=COLORS['gap'], fontweight='bold')

    ax.set_title('An evolutionary game of plant light competition: from molecular light signals to dense-planting breeding',
                  fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 2: ESS vs population optimum across density
# ============================================================
def plot_ess_vs_optimum(p: GameParams, save_path: str,
                          density_range: Tuple[float, float, int] = (10, 5000, 200)):
    N_min, N_max, n_points = density_range
    densities = np.logspace(np.log10(N_min), np.log10(N_max), n_points)

    ess_list, opt_list, yloss = [], [], []
    for N in densities:
        ess = find_ess(N, p)
        s_opt = find_population_optimum(N, p)
        yl = yield_loss_percentage(N, p)
        ess_list.append(ess if ess is not None else np.nan)
        opt_list.append(s_opt if s_opt is not None else np.nan)
        yloss.append(yl if yl is not None else np.nan)
    ess_arr, opt_arr, yloss_arr = map(np.array, (ess_list, opt_list, yloss))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(densities, ess_arr, color=COLORS['ess'], linewidth=2.5,
             label=r'Evolutionarily stable strategy ($s^{*}$)')
    ax1.plot(densities, opt_arr, color=COLORS['opt'], linewidth=2.5, linestyle='--',
             label=r'Population optimum ($s_{opt}$)')
    ax1.fill_between(densities, opt_arr, ess_arr, alpha=0.15, color=COLORS['gap'],
                      label='Tragedy gap')
    for N_label, name in [(100, 'sparse'), (400, 'medium'), (1000, 'dense')]:
        ax1.axvline(x=N_label, color='gray', linestyle=':', alpha=0.5)
        ax1.text(N_label, 0.02, name, ha='center', va='bottom', fontsize=9, color='gray')

    ax1.set_xscale('log')
    ax1.set_xlabel(DEN_LABEL)
    ax1.set_ylabel(STRAT_LABEL)
    ax1.set_title('(a) ESS vs population optimum', fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)

    gap = ess_arr - opt_arr
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(densities, gap, color=COLORS['gap'], linewidth=2.5,
                   label=r'Strategy gap, $\Delta s = s^{*} - s_{opt}$')
    l2 = ax2_twin.plot(densities, yloss_arr, color=COLORS['ess'], linewidth=2.5,
                         linestyle='--', label='Yield loss (%)')

    ax2.set_xscale('log')
    ax2.set_xlabel(DEN_LABEL)
    ax2.set_ylabel(r'Strategy gap, $\Delta s$', color=COLORS['gap'])
    ax2_twin.set_ylabel('Yield loss (%)', color=COLORS['ess'])
    ax2.set_title('(b) Tragedy gap and yield loss', fontweight='bold')
    lines = l1 + l2
    ax2.legend(lines, [l.get_label() for l in lines], loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Evolutionarily stable strategy vs population optimum across planting density',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 3: evolutionary convergence trajectories
# ============================================================
def plot_evolution_dynamics(p: GameParams, save_path: str,
                              densities: List[float] = None,
                              generations: int = 300):
    if densities is None:
        densities = [100, 400, 1000, 3000]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for idx, N in enumerate(densities):
        ax = axes[idx]
        sim = EvolutionSimulation(N, p, pop_size=500, initial_mean=0.5)
        result = sim.run(generations=generations)

        generations_arr = np.arange(1, len(result['history_mean']) + 1)
        ess = result['ess_predicted']
        s_opt = result['population_optimum']

        ax.plot(generations_arr, result['history_mean'], color=COLORS['density'],
                linewidth=2, label='Population mean strategy')
        ax.fill_between(generations_arr,
                         np.array(result['history_mean']) - np.array(result['history_std']),
                         np.array(result['history_mean']) + np.array(result['history_std']),
                         alpha=0.2, color=COLORS['density'])

        if ess is not None:
            ax.axhline(y=ess, color=COLORS['ess'], linestyle='--', linewidth=2,
                        label=f'ESS prediction = {ess:.3f}')
        if s_opt is not None:
            ax.axhline(y=s_opt, color=COLORS['opt'], linestyle=':', linewidth=2,
                        label=f'Population optimum = {s_opt:.3f}')

        ax.set_xlabel('Generation')
        ax.set_ylabel(STRAT_LABEL)
        ax.set_title(f'$N$ = {N} effective neighbours', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Evolutionary convergence to the ESS at different densities (individual-based model)',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 4: pairwise invasibility plot
# ============================================================
def plot_pip(p: GameParams, save_path: str, N: float = 500):
    s_grid = np.linspace(0.01, 0.99, 100)
    S_resident, S_mutant = np.meshgrid(s_grid, s_grid)

    invasion = np.zeros_like(S_resident)
    for i in range(len(s_grid)):
        for j in range(len(s_grid)):
            invasion[i, j] = pairwise_invasibility(S_mutant[i, j], S_resident[i, j], N, p)

    ess = find_ess(N, p)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    vmax = np.percentile(np.abs(invasion), 95)
    im = ax.pcolormesh(S_resident, S_mutant, invasion, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, label=r'Invasion index, $\log(W_{mut}/W_{res})$')

    ax.contour(S_resident, S_mutant, invasion, levels=[0], colors='black',
                linewidths=2, linestyles='-')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    if ess is not None:
        ax.plot(ess, ess, marker='*', markersize=18, color='yellow', markeredgecolor='black',
                linestyle='None', label=f'ESS = {ess:.3f}')
        ax.axvline(x=ess, color='yellow', linestyle=':', alpha=0.7)
        ax.axhline(y=ess, color='yellow', linestyle=':', alpha=0.7)

    s_opt = find_population_optimum(N, p)
    if s_opt is not None:
        ax.plot(s_opt, s_opt, 'go', markersize=12, markeredgecolor='black',
                label=f'$s_{{opt}}$ = {s_opt:.3f}')

    ax.set_xlabel(r'Resident strategy, $s_{res}$')
    ax.set_ylabel(r'Mutant strategy, $s_{mut}$')
    ax.set_title(f'Pairwise invasibility plot, $N$ = {N} effective neighbours', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 5: frequency-dependent competition among mixed strategies
# ============================================================
def plot_mixed_competition(p: GameParams, save_path: str, N: float = 500):
    strategies = [0.2, 0.4, 0.6, 0.8]
    result = mixed_strategy_competition(N, p, strategies=strategies, generations=150)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i, s in enumerate(strategies):
        ax1.plot(result['frequency_history'][:, i], color=colors[i], linewidth=2.5,
                  label=f'$s$ = {s}')

    ess = result['ess_predicted']
    if ess is not None:
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        dom = strategies[int(np.argmax(result['final_frequencies']))]
        ax1.text(0.04, 0.96, f'ESS prediction = {ess:.3f}\nwinning strategy: $s$ = {dom}',
                  transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Genotype frequency')
    ax1.set_title('(a) Frequency dynamics in mixed stands', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)

    densities_test = [100, 300, 500, 1000, 2000]
    final_freq_matrix = []
    for N_test in densities_test:
        res = mixed_strategy_competition(N_test, p, strategies=strategies, generations=150)
        final_freq_matrix.append(res['final_frequencies'])
    final_freq_matrix = np.array(final_freq_matrix)

    x = np.arange(len(densities_test))
    width = 0.18
    for i, s in enumerate(strategies):
        ax2.bar(x + i * width, final_freq_matrix[:, i], width, color=colors[i],
                label=f'$s$ = {s}')

    ax2.set_xlabel(DEN_LABEL)
    ax2.set_ylabel('Final genotype frequency')
    ax2.set_title('(b) Competition outcome across densities', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([str(d) for d in densities_test])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Frequency-dependent competition among shade-avoidance strategies',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 6: breeding strategy optimization
# ============================================================
def plot_breeding_strategy(p: GameParams, save_path: str):
    densities = np.logspace(np.log10(50), np.log10(3000), 100)

    reduction_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ['No attenuation (ESS)', '25% attenuation', '50% attenuation',
              '75% attenuation', 'Full attenuation ($s_{opt}$)']
    colors_breed = ['#D62728', '#FF7F0E', '#FFC107', '#8BC34A', '#2CA02C']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for rr, label, color in zip(reduction_ratios, labels, colors_breed):
        targets = []
        for N in densities:
            bt = breeding_target(N, p, reduction_ratio=rr)
            targets.append(bt.get('breeding_target_strategy', np.nan))
        ax1.plot(densities, targets, color=color, linewidth=2, label=label)

    ax1.set_xscale('log')
    ax1.set_xlabel(DEN_LABEL)
    ax1.set_ylabel(r'Breeding target strategy, $s$')
    ax1.set_title('(a) Breeding target for different attenuation levels', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)

    for rr, label, color in zip(reduction_ratios[1:], labels[1:], colors_breed[1:]):
        gains = []
        for N in densities:
            bt = breeding_target(N, p, reduction_ratio=rr)
            gains.append(bt.get('expected_yield_gain_percent', np.nan))
        ax2.plot(densities, gains, color=color, linewidth=2, label=label)

    # Generic density reference bands (no crop-specific absolute densities; see Methods scale note)
    for N_label, name in [(100, 'sparse'), (400, 'medium'), (1000, 'dense')]:
        ax2.axvline(x=N_label, color='gray', linestyle=':', alpha=0.4)
        ax2.text(N_label, 0.4, name, ha='center', va='bottom', fontsize=8, color='gray',
                 rotation=90)

    ax2.set_xscale('log')
    ax2.set_xlabel(DEN_LABEL)
    ax2.set_ylabel('Expected yield gain (%)')
    ax2.set_title('(b) Expected yield gain of attenuation', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Breeding strategy optimization: targets and expected yield gains under dense planting',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Fig. 7: light signalling (phyB) coupling to phenotype and yield
# ============================================================
def plot_light_signaling_coupling(p: GameParams, save_path: str, N: float = 500):
    phyB_range = np.linspace(0.1, 1.5, 50)
    ess_base = find_ess(N, p)

    phenotype_strategies, yields = [], []
    for phyB_s in phyB_range:
        s_pheno = light_signaling_coupling(ess_base if ess_base else 0.5,
                                             phyB_sensitivity=phyB_s)
        phenotype_strategies.append(s_pheno)
        w = fitness(s_pheno, s_pheno, N, p)
        yields.append(w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(phyB_range, phenotype_strategies, color=COLORS['ess'], linewidth=2.5)
    ax1.axvline(x=1.0, color='gray', linestyle='--', label='Wild-type phyB')
    ax1.axhline(y=find_population_optimum(N, p), color=COLORS['opt'], linestyle=':',
                label='Population optimum')
    ax1.set_xlabel('phyB pathway sensitivity (relative)')
    ax1.set_ylabel(STRAT_LABEL)
    ax1.set_title('(a) phyB modulation of the avoidance phenotype', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    w_wt = fitness(ess_base if ess_base else 0.5, ess_base if ess_base else 0.5, N, p)
    yield_percent = [100 * (y - w_wt) / w_wt for y in yields]
    ax2.plot(phyB_range, yield_percent, color=COLORS['opt'], linewidth=2.5)
    ax2.axvline(x=1.0, color='gray', linestyle='--', label='Wild-type phyB')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.fill_between(phyB_range, 0, yield_percent,
                       where=np.array(yield_percent) > 0, alpha=0.2, color='green')
    ax2.fill_between(phyB_range, 0, yield_percent,
                       where=np.array(yield_percent) < 0, alpha=0.2, color='red')
    ax2.set_xlabel('phyB pathway sensitivity (relative)')
    ax2.set_ylabel('Yield change (%)')
    ax2.set_title('(b) Yield effect of phyB attenuation at high density', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    opt_idx = int(np.argmax(yield_percent))
    ax2.annotate(f'optimum phyB ≈ {phyB_range[opt_idx]:.2f}\nyield gain ≈ {yield_percent[opt_idx]:.1f}%',
                  xy=(phyB_range[opt_idx], yield_percent[opt_idx]),
                  xytext=(phyB_range[opt_idx] + 0.15, yield_percent[opt_idx] - 2.5),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle(f'Coupling of phyB light signalling to phenotype and dense-stand yield ($N$ = {N})',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  saved: {save_path}")


# ============================================================
# Generate all figures
# ============================================================
def generate_all_figures(p: GameParams, output_dir: str):
    import os
    os.makedirs(output_dir, exist_ok=True)
    setup_style()
    print("Generating figures...")
    plot_model_framework(p, f"{output_dir}/Fig1_model_framework.png")
    plot_ess_vs_optimum(p, f"{output_dir}/Fig2_ess_vs_optimum.png")
    plot_evolution_dynamics(p, f"{output_dir}/Fig3_evolution_dynamics.png")
    plot_pip(p, f"{output_dir}/Fig4_PIP.png")
    plot_mixed_competition(p, f"{output_dir}/Fig5_mixed_competition.png")
    plot_breeding_strategy(p, f"{output_dir}/Fig6_breeding_strategy.png")
    plot_light_signaling_coupling(p, f"{output_dir}/Fig7_light_signaling.png")
    print("All figures generated.")


if __name__ == '__main__':
    p = GameParams()
    generate_all_figures(p, '../figures')
