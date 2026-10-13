#!/usr/bin/env python3
"""
主运行脚本：植物光竞争进化博弈模型与密植育种策略优化

运行方式:
    cd src
    python main.py

输出:
    - figures/: 论文图表 (Fig1-Fig7)
    - data/: 数值结果CSV
    - 控制台输出核心结论
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (GameParams, find_ess, find_population_optimum, tragedy_gap,
                    yield_loss_percentage, breeding_target, light_signaling_coupling)
from simulation import EvolutionSimulation, run_density_comparison
from visualization import generate_all_figures, setup_style
from benchmark import run_all_benchmarks
from molecular import run_molecular_analysis
from empirical_validation import run_empirical_validation
from sensitivity import run_sensitivity


def run_core_analysis(p: GameParams, output_dir: str):
    """运行核心分析，输出数值结果"""
    print("=" * 70)
    print("核心分析：不同密度下的ESS、群体最优与公地悲剧")
    print("=" * 70)

    # 密度梯度
    densities = [10, 25, 50, 100, 200, 300, 400, 500, 600, 800,
                 1000, 1500, 2000, 3000, 5000]

    results = []
    for N in densities:
        ess, s_opt, gap = tragedy_gap(N, p)
        yloss = yield_loss_percentage(N, p)
        results.append({
            'density': N,
            'ESS': round(ess, 4) if ess else None,
            'population_optimum': round(s_opt, 4) if s_opt else None,
            'tragedy_gap': round(gap, 4) if gap else None,
            'yield_loss_percent': round(yloss, 2) if yloss else None,
        })
        print(f"  N={N:5d}: ESS={ess:.4f}, s_opt={s_opt:.4f}, "
              f"gap={gap:.4f}, 产量损失={yloss:.2f}%")

    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/core_analysis_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  核心结果已保存: {csv_path}")

    return df


def run_breeding_analysis(p: GameParams, output_dir: str):
    """运行育种策略分析"""
    print("\n" + "=" * 70)
    print("育种策略分析：不同密度下的育种目标与预期产量增益")
    print("=" * 70)

    crop_densities = {
        '水稻_常规': 250,
        '水稻_密植': 500,
        '小麦_常规': 400,
        '小麦_密植': 800,
        '玉米_常规': 8,
        '玉米_密植': 12,
    }

    breeding_results = []
    for crop_name, N in crop_densities.items():
        print(f"\n  --- {crop_name} (N={N}) ---")
        for rr in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bt = breeding_target(N, p, reduction_ratio=rr)
            breeding_results.append({
                'crop': crop_name,
                'density': N,
                'reduction_ratio': rr,
                'ESS': round(bt.get('ess', 0), 4),
                'population_optimum': round(bt.get('population_optimum', 0), 4),
                'breeding_target': round(bt.get('breeding_target_strategy', 0), 4),
                'expected_yield_gain_percent': round(bt.get('expected_yield_gain_percent', 0), 2),
            })
            label = ['不改良', '25%改良', '50%改良', '75%改良', '完全改良'][int(rr * 4)]
            print(f"    {label}: 目标策略={bt['breeding_target_strategy']:.4f}, "
                  f"预期增产={bt['expected_yield_gain_percent']:.2f}%")

    df_breed = pd.DataFrame(breeding_results)
    csv_path = f"{output_dir}/breeding_strategy_results.csv"
    df_breed.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  育种结果已保存: {csv_path}")

    return df_breed


def run_evolution_validation(p: GameParams, output_dir: str):
    """运行演化模拟验证ESS收敛性"""
    print("\n" + "=" * 70)
    print("演化模拟验证：不同密度下ESS的收敛性")
    print("=" * 70)

    test_densities = [100, 400, 1000, 3000]
    validation_results = []

    for N in test_densities:
        print(f"\n  运行密度 N={N}...")
        sim = EvolutionSimulation(N, p, pop_size=500, initial_mean=0.5)
        result = sim.run(generations=300)
        validation_results.append({
            'density': N,
            'ESS_predicted': round(result['ess_predicted'], 4),
            'simulated_final_mean': round(result['final_mean'], 4),
            'convergence_error': round(abs(result['final_mean'] - result['ess_predicted']), 4),
            'converged': result['converged_to_ess'],
        })
        print(f"    ESS预测={result['ess_predicted']:.4f}, "
              f"模拟收敛={result['final_mean']:.4f}, "
              f"误差={abs(result['final_mean'] - result['ess_predicted']):.4f}, "
              f"收敛={'是' if result['converged_to_ess'] else '否'}")

    df_val = pd.DataFrame(validation_results)
    csv_path = f"{output_dir}/evolution_validation_results.csv"
    df_val.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  验证结果已保存: {csv_path}")

    return df_val


def run_light_signaling_analysis(p: GameParams, output_dir: str):
    """光信号通路耦合分析"""
    print("\n" + "=" * 70)
    print("光信号通路耦合分析：phyB敏感性对表型与产量的影响")
    print("=" * 70)

    N = 500
    ess_base = find_ess(N, p)
    print(f"  基准密度 N={N}, 野生型ESS={ess_base:.4f}")

    phyB_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    signaling_results = []

    for phyB_s in phyB_values:
        s_pheno = light_signaling_coupling(ess_base, phyB_sensitivity=phyB_s)
        from model import fitness
        w = fitness(s_pheno, s_pheno, N, p)
        w_wt = fitness(ess_base, ess_base, N, p)
        yield_change = 100 * (w - w_wt) / w_wt
        signaling_results.append({
            'phyB_sensitivity': phyB_s,
            'phenotype_strategy': round(s_pheno, 4),
            'yield_change_percent': round(yield_change, 2),
        })
        print(f"    phyB={phyB_s:.1f}: 表型策略={s_pheno:.4f}, 产量变化={yield_change:+.2f}%")

    df_sig = pd.DataFrame(signaling_results)
    csv_path = f"{output_dir}/light_signaling_results.csv"
    df_sig.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  光信号结果已保存: {csv_path}")

    return df_sig


def print_summary(p: GameParams):
    """打印核心结论摘要"""
    print("\n" + "=" * 70)
    print("核心结论摘要")
    print("=" * 70)

    # 关键密度点
    key_densities = [100, 250, 400, 500, 800, 1000]
    print("\n  公地悲剧的核心证据：ESS > 群体最优 → 过度避荫 → 密植减产")
    print("  " + "-" * 60)
    for N in key_densities:
        ess, s_opt, gap = tragedy_gap(N, p)
        yloss = yield_loss_percentage(N, p)
        print(f"  N={N:4d}: ESS={ess:.3f} > s_opt={s_opt:.3f}, "
              f"偏差={gap:.3f}, 产量损失={yloss:.1f}%")

    print("\n  育种启示：")
    print("  1. 自然选择演化出的避荫强度(ESS)高于群体最优(s_opt)，形成公地悲剧")
    print("  2. 密度越高，偏差越大，密植减产越严重")
    print("  3. 弱化phyB通路敏感性（避荫弱敏感株系）可缩小偏差，提升密植产量")
    print("  4. 育种目标：将光响应强度从ESS向s_opt方向改良，幅度取决于目标密度")
    print("  5. 这解释了为什么现代作物仍保留过度避荫——驯化未完成光竞争优化")


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figure_dir = f"{base_dir}/figures"
    data_dir = f"{base_dir}/data"
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("植物光竞争进化博弈模型与密植育种策略优化")
    print("目标期刊: New Phytologist")
    print()

    # 初始化模型参数
    p = GameParams()
    print(f"模型参数: k0={p.k0}, k1={p.k1}, gamma={p.gamma}, "
          f"c={p.c}, alpha={p.alpha}, beta={p.beta}")
    print()

    # 运行各项分析
    df_core = run_core_analysis(p, data_dir)
    df_breed = run_breeding_analysis(p, data_dir)
    df_val = run_evolution_validation(p, data_dir)
    df_sig = run_light_signaling_analysis(p, data_dir)

    # 生成图表
    print("\n" + "=" * 70)
    print("生成论文图表...")
    print("=" * 70)
    generate_all_figures(p, figure_dir)

    # 文献对标验证（复现 Gersani2001 / Bongers2019 / Burbano2025 / 公地悲剧）
    bench = run_all_benchmarks(p, base_dir)

    # 分子-表型-群体跨层耦合（phyB/CRY/PIF/auxin -> 策略 s -> 产量）
    molec = run_molecular_analysis(p, base_dir)

    # 独立田间数据实证反演（Burbano2025 IL2-2 产量点反演增益 g 与悲剧损失 L）
    empir, empir_fig = run_empirical_validation(base_dir, p)

    # 全局参数敏感性与稳健性（Monte Carlo + OAT）
    sens = run_sensitivity(base_dir)

    # 打印结论
    print_summary(p)

    print("\n" + "=" * 70)
    print("全部分析完成！")
    print(f"  图表目录: {figure_dir}")
    print(f"  数据目录: {data_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
