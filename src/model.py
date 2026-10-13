"""
核心博弈模型：植物光竞争的进化博弈论框架

核心概念：
- 策略 s：光响应强度（避荫反应强度），连续值 [0, 1]
  s=0: 完全不避荫（耐荫型）；s=1: 最强避荫反应
- 适合度 W：个体光合收益 - 构建成本
- ESS：演化稳定策略，自然选择的终点
- s_opt：群体最优策略，总产量最大化的策略
- 公地悲剧偏差：ESS - s_opt > 0，即自然选择导致过度避荫

模型设计参考：
- Falster & Westoby (2003) Plant height and evolutionary games
- Gersani et al. (2001) Tragedy of the commons in root competition
- NP (2014) Crown depth as a result of evolutionary games
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GameParams:
    """博弈模型参数容器"""
    k0: float = 1.0       # 基础光合收益（不依赖策略）
    k1: float = 0.012     # 相对高度竞争收益系数（乘以N^beta，密度越高竞争越激烈）
    gamma: float = 0.2    # 绝对高度的光捕获增益系数
    c: float = 0.3        # 单位策略的构建成本系数
    alpha: float = 2.0    # 成本非线性指数 (>1)
    beta: float = 0.5     # 密度效应系数（竞争强度随密度增长）
    s_min: float = 0.0    # 策略下界
    s_max: float = 1.0    # 策略上界


def fitness(s_i: float, s_bar: float, N: float, p: GameParams) -> float:
    """
    个体适合度函数

    参数:
        s_i: 个体自身的光响应强度（策略）
        s_bar: 种群平均光响应强度
        N: 种群密度（plants/m^2）
        p: 模型参数

    返回:
        个体适合度（相对值）

    公式分解：
        W = 基础光合 + 绝对高度增益 + 相对高度竞争收益 - 构建成本
        基础光合: k0（所有个体共享的基础光捕获）
        绝对高度增益: k0 * gamma * s_i（即使均匀种群，更高也有微弱收益）
        相对高度竞争收益: k1 * (s_i - s_bar) * N^beta
            - 当 s_i > s_bar 时，个体高于平均，获得正收益（截获更多光）
            - 当 s_i < s_bar 时，个体低于平均，遭受损失（被遮荫）
            - 当 s_i = s_bar 时，相对优势消失，竞争收益为0
            - 密度N越高，竞争收益/损失的幅度越大（N^beta）
        构建成本: c * s_i^alpha（非线性递增，alpha>1）

    生物学意义：
        密植条件下，邻居更近，遮荫效应更强，因此相对高度优势的价值更大，
        自然选择会 favor 更强烈的避荫反应（ESS随密度增加而增加）。
    """
    # 密度效应：N^beta，密度越高竞争越激烈
    density_factor = N ** p.beta

    # 基础光合收益（绝对高度带来的微弱增益）
    base_gain = p.k0 * (1.0 + p.gamma * s_i)

    # 相对高度竞争收益（频率依赖核心项）
    # 个体与种群平均的高度差决定竞争收益/损失
    competitive_gain = p.k1 * (s_i - s_bar) * density_factor

    # 构建成本（非线性递增）
    construction_cost = p.c * (s_i ** p.alpha)

    return base_gain + competitive_gain - construction_cost


def fitness_derivative(s_i: float, s_bar: float, N: float, p: GameParams) -> float:
    """
    适合度对个体策略 s_i 的偏导数（用于ESS解析条件）

    ∂W/∂s_i = k0*gamma + k1*N^beta - c*alpha*s_i^(alpha-1)
    """
    density_factor = N ** p.beta
    d_base = p.k0 * p.gamma
    d_competitive = p.k1 * density_factor
    d_cost = p.c * p.alpha * (s_i ** (p.alpha - 1.0))
    return d_base + d_competitive - d_cost


def find_ess(N: float, p: GameParams) -> Optional[float]:
    """
    求解演化稳定策略（ESS）—— 解析解

    ESS条件：当种群全部采用 s* 时，任何突变策略的适合度均不高于 s*
    数学条件：∂W/∂s_i |_{s_i=s_bar=s*} = 0 （一阶条件）
              且 ∂²W/∂s_i² < 0 （二阶条件，自动满足因为alpha>1）

    解析推导：
        ∂W/∂s_i = k0*gamma + k1*N^beta - c*alpha*s_i^(alpha-1) = 0
        => s* = [(k0*gamma + k1*N^beta) / (c*alpha)]^(1/(alpha-1))

    关键性质：ESS随密度N增加而增加（密植→更强烈避荫），符合生物学直觉。

    参数:
        N: 种群密度
        p: 模型参数

    返回:
        ESS策略值 s*（clip到[s_min, s_max]）
    """
    if p.alpha <= 1.0:
        return p.s_max

    numerator = p.k0 * p.gamma + p.k1 * (N ** p.beta)
    denominator = p.c * p.alpha

    if numerator <= 0:
        return p.s_min

    ess = (numerator / denominator) ** (1.0 / (p.alpha - 1.0))
    return np.clip(ess, p.s_min, p.s_max)


def find_population_optimum(N: float, p: GameParams) -> Optional[float]:
    """
    求解群体最优策略（总产量最大化）—— 解析解

    当种群全部采用策略 s 时，s_i = s_bar = s，竞争收益项 (s_i - s_bar) = 0：
    W(s,s,N) = k0*(1+gamma*s) - c*s^alpha
    总适合度 W_total = N * W(s,s,N)

    对s求导: dW_total/ds = N * [k0*gamma - c*alpha*s^(alpha-1)] = 0
    解析解: s_opt = [k0*gamma / (c*alpha)]^(1/(alpha-1))

    关键性质：群体最优与密度N无关！因为均匀种群中相对高度优势消失，
    竞争收益为零，只有绝对高度增益和构建成本的权衡。
    这正是公地悲剧的根源：个体理性（ESS考虑竞争收益）导致群体非理性。

    参数:
        N: 种群密度（实际上不影响s_opt，保留参数用于接口一致）
        p: 模型参数

    返回:
        群体最优策略值 s_opt
    """
    if p.alpha <= 1.0:
        return p.s_min

    numerator = p.k0 * p.gamma
    denominator = p.c * p.alpha

    if numerator <= 0:
        return p.s_min

    s_opt = (numerator / denominator) ** (1.0 / (p.alpha - 1.0))
    return np.clip(s_opt, p.s_min, p.s_max)


def tragedy_gap(N: float, p: GameParams) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    计算公地悲剧偏差（核心指标）

    返回:
        (ess, s_opt, gap)
        gap = ess - s_opt > 0 表示过度避荫（公地悲剧）
        gap 越大，密植减产越严重
    """
    ess = find_ess(N, p)
    s_opt = find_population_optimum(N, p)
    if ess is not None and s_opt is not None:
        gap = ess - s_opt
        return ess, s_opt, gap
    return None, None, None


def yield_loss_percentage(N: float, p: GameParams) -> Optional[float]:
    """
    计算公地悲剧导致的产量损失百分比

    产量损失 = 100 * (W_opt - W_ess) / W_opt
    其中 W_opt 是群体最优时的单株适合度，W_ess 是ESS时的单株适合度
    """
    ess, s_opt, gap = tragedy_gap(N, p)
    if ess is None or s_opt is None:
        return None

    w_ess = fitness(ess, ess, N, p)
    w_opt = fitness(s_opt, s_opt, N, p)

    if w_opt > 1e-10:
        return 100.0 * (w_opt - w_ess) / w_opt
    return None


def pairwise_invasibility(s_mutant: float, s_resident: float, N: float, p: GameParams) -> float:
    """
    成对入侵性分析（PIP）

    计算突变体在常驻种群中的入侵指数：
    invasion_exponent = log(W_mutant / W_resident)

    > 0: 突变体可以入侵
    < 0: 突变体被排斥
    = 0: 中性

    参数:
        s_mutant: 突变体策略
        s_resident: 常驻种群策略
        N: 密度
        p: 参数
    """
    w_mutant = fitness(s_mutant, s_resident, N, p)
    w_resident = fitness(s_resident, s_resident, N, p)

    if w_resident > 1e-10 and w_mutant > 1e-10:
        return np.log(w_mutant / w_resident)
    return -np.inf


def light_signaling_coupling(s_base: float, phyB_sensitivity: float = 1.0,
                              CRY_sensitivity: float = 0.6, PIF_activity: float = 0.8) -> float:
    """
    光信号通路耦合：将分子通路参数映射到表型策略

    生物学逻辑：
    - phyB感知R/FR降低，是避荫反应的主调控因子
    - CRY感知蓝光，辅助调控避荫反应
    - PIFs是下游转录因子，整合光信号与激素信号
    - 光响应强度 s = s_base * (w1*phyB + w2*CRY + w3*PIF)

    参数:
        s_base: 基础策略值（遗传背景决定）
        phyB_sensitivity: phyB通路敏感性（1.0=野生型，<1=弱化）
        CRY_sensitivity: CRY通路敏感性
        PIF_activity: PIFs转录活性

    返回:
        耦合后的表型策略值
    """
    w_phyB = 0.5   # phyB主导
    w_CRY = 0.2     # CRY辅助
    w_PIF = 0.3     # PIF整合

    signaling_factor = (w_phyB * phyB_sensitivity +
                        w_CRY * CRY_sensitivity +
                        w_PIF * PIF_activity)
    return np.clip(s_base * signaling_factor, 0.0, 1.0)


def breeding_target(density: float, p: GameParams, reduction_ratio: float = 0.0) -> dict:
    """
    输出育种目标策略

    参数:
        density: 目标种植密度
        p: 模型参数
        reduction_ratio: 育种目标相对于ESS的降低比例
                        0 = 保持ESS（不改良）
                        1 = 完全达到群体最优（理想极限）

    返回:
        包含育种目标、预期产量增益等信息的字典
    """
    ess, s_opt, gap = tragedy_gap(density, p)
    if ess is None:
        return {}

    # 育种目标策略
    s_breed = ess - reduction_ratio * gap
    s_breed = np.clip(s_breed, p.s_min, p.s_max)

    # 预期产量
    w_ess = fitness(ess, ess, density, p)
    w_breed = fitness(s_breed, s_breed, density, p)
    yield_gain = 100.0 * (w_breed - w_ess) / w_ess if w_ess > 1e-10 else 0.0

    return {
        'density': density,
        'ess': ess,
        'population_optimum': s_opt,
        'tragedy_gap': gap,
        'reduction_ratio': reduction_ratio,
        'breeding_target_strategy': s_breed,
        'expected_yield_gain_percent': yield_gain,
        'yield_loss_at_ess_percent': yield_loss_percentage(density, p),
    }


if __name__ == '__main__':
    # 快速测试
    p = GameParams()
    for N in [50, 200, 500, 1000, 3000]:
        ess, s_opt, gap = tragedy_gap(N, p)
        yloss = yield_loss_percentage(N, p)
        print(f"N={N:5d}: ESS={ess:.4f}, s_opt={s_opt:.4f}, gap={gap:.4f}, yield_loss={yloss:.2f}%")
