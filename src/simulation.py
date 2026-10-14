"""
演化模拟模块：基于个体的模型（IBM）与演化动态模拟

验证ESS的收敛性，模拟不同初始条件下的演化轨迹
"""

import numpy as np
from typing import List, Tuple, Optional
from model import GameParams, fitness, find_ess, find_population_optimum


class EvolutionSimulation:
    """
    基于个体的演化模拟

    每一代：
    1. 计算每个个体的适合度（基于自身策略与种群平均策略）
    2. 按适合度比例进行繁殖（轮盘赌选择）
    3. 后代产生突变（高斯扰动）
    4. 记录种群平均策略的演化轨迹
    """

    def __init__(self, N: float, p: GameParams,
                 pop_size: int = 1000,
                 mutation_rate: float = 0.01,
                 mutation_sd: float = 0.02,
                 initial_mean: float = 0.5,
                 initial_sd: float = 0.1,
                 seed: int = 42):
        self.N = N
        self.p = p
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_sd = mutation_sd

        # Dedicated, seeded RNG so every run is exactly reproducible
        # (independent of the global numpy random state).
        self.rng = np.random.default_rng(seed)

        # 初始化种群策略
        self.strategies = self.rng.normal(initial_mean, initial_sd, pop_size)
        self.strategies = np.clip(self.strategies, p.s_min, p.s_max)

        # 记录历史
        self.history_mean = []
        self.history_std = []
        self.history_fitness = []

    def step(self) -> Tuple[float, float, float]:
        """
        执行一代演化

        返回: (种群平均策略, 策略标准差, 平均适合度)
        """
        s_bar = np.mean(self.strategies)

        # 计算每个个体的适合度
        fitnesses = np.array([
            fitness(s_i, s_bar, self.N, self.p)
            for s_i in self.strategies
        ])

        # 确保适合度为正（用于轮盘赌选择）
        fitnesses_pos = fitnesses - np.min(fitnesses) + 1e-6
        total_fitness = np.sum(fitnesses_pos)

        if total_fitness < 1e-10:
            # 所有个体适合度极低，随机保留
            parents = self.rng.integers(0, self.pop_size, self.pop_size)
        else:
            # 轮盘赌选择
            probabilities = fitnesses_pos / total_fitness
            parents = self.rng.choice(self.pop_size, size=self.pop_size,
                                      replace=True, p=probabilities)

        # 产生后代 + 突变
        offspring = self.strategies[parents].copy()
        mutation_mask = self.rng.random(self.pop_size) < self.mutation_rate
        mutations = self.rng.normal(0, self.mutation_sd, self.pop_size)
        offspring[mutation_mask] += mutations[mutation_mask]
        offspring = np.clip(offspring, self.p.s_min, self.p.s_max)

        self.strategies = offspring

        mean_s = np.mean(self.strategies)
        std_s = np.std(self.strategies)
        mean_fitness = np.mean(fitnesses)

        self.history_mean.append(mean_s)
        self.history_std.append(std_s)
        self.history_fitness.append(mean_fitness)

        return mean_s, std_s, mean_fitness

    def run(self, generations: int = 500, verbose: bool = False) -> dict:
        """
        运行多代演化

        返回:
            包含演化轨迹、收敛状态等信息的字典
        """
        for gen in range(generations):
            mean_s, std_s, mean_f = self.step()
            if verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1}/{generations}: mean_s={mean_s:.4f}, "
                      f"std={std_s:.4f}, fitness={mean_f:.4f}")

        ess = find_ess(self.N, self.p)
        s_opt = find_population_optimum(self.N, self.p)

        return {
            'N': self.N,
            'generations': generations,
            'final_mean': np.mean(self.strategies),
            'final_std': np.std(self.strategies),
            'final_fitness': np.mean(self.history_fitness[-10:]),
            'ess_predicted': ess,
            'population_optimum': s_opt,
            'converged_to_ess': abs(np.mean(self.strategies) - ess) < 0.05 if ess else False,
            'history_mean': np.array(self.history_mean),
            'history_std': np.array(self.history_std),
            'history_fitness': np.array(self.history_fitness),
        }


def run_density_comparison(densities: List[float], p: GameParams,
                            generations: int = 300, pop_size: int = 500) -> List[dict]:
    """
    在不同密度下运行演化模拟，比较收敛结果

    参数:
        densities: 密度列表
        p: 模型参数
        generations: 演化代数
        pop_size: 种群大小

    返回:
        每个密度的演化结果列表
    """
    results = []
    for N in densities:
        print(f"运行密度 N={N}...")
        sim = EvolutionSimulation(N, p, pop_size=pop_size)
        result = sim.run(generations=generations)
        results.append(result)
        print(f"  收敛: mean={result['final_mean']:.4f}, "
              f"ESS预测={result['ess_predicted']:.4f}, "
              f"收敛={result['converged_to_ess']}")
    return results


def run_multiple_initial_conditions(N: float, p: GameParams,
                                      initial_means: List[float] = None,
                                      generations: int = 300) -> List[dict]:
    """
    不同初始条件下的演化模拟，验证ESS的全局收敛性

    参数:
        N: 密度
        p: 参数
        initial_means: 初始策略均值列表
        generations: 代数
    """
    if initial_means is None:
        initial_means = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for init_mean in initial_means:
        sim = EvolutionSimulation(N, p, initial_mean=init_mean, initial_sd=0.05)
        result = sim.run(generations=generations)
        result['initial_mean'] = init_mean
        results.append(result)
    return results


def mixed_strategy_competition(N: float, p: GameParams,
                                strategies: List[float],
                                initial_frequencies: List[float] = None,
                                generations: int = 200) -> dict:
    """
    多策略混合竞争模拟：追踪不同策略基因型的频率变化

    参数:
        N: 密度
        p: 参数
        strategies: 策略值列表（如 [0.2, 0.5, 0.8]）
        initial_frequencies: 初始频率，默认均匀
        generations: 代数

    返回:
        频率演化轨迹
    """
    n_strategies = len(strategies)
    if initial_frequencies is None:
        frequencies = np.ones(n_strategies) / n_strategies
    else:
        frequencies = np.array(initial_frequencies, dtype=float)
        frequencies /= frequencies.sum()

    frequency_history = [frequencies.copy()]

    for gen in range(generations):
        # 计算种群平均策略
        s_bar = np.sum(frequencies * np.array(strategies))

        # 计算各策略的适合度
        fitnesses = np.array([fitness(s, s_bar, N, p) for s in strategies])
        fitnesses_pos = fitnesses - np.min(fitnesses) + 1e-6

        # 更新频率（复制者方程）
        mean_fitness = np.sum(frequencies * fitnesses_pos)
        if mean_fitness > 1e-10:
            frequencies = frequencies * fitnesses_pos / mean_fitness
            frequencies /= frequencies.sum()

        frequency_history.append(frequencies.copy())

    return {
        'N': N,
        'strategies': strategies,
        'frequency_history': np.array(frequency_history),
        'final_frequencies': frequencies,
        'ess_predicted': find_ess(N, p),
    }


if __name__ == '__main__':
    from model import GameParams
    p = GameParams()

    # 测试单密度演化
    print("=== 单密度演化测试 (N=500) ===")
    sim = EvolutionSimulation(500, p, pop_size=500)
    result = sim.run(generations=200, verbose=True)
    print(f"最终均值: {result['final_mean']:.4f}")
    print(f"ESS预测: {result['ess_predicted']:.4f}")
    print(f"收敛: {result['converged_to_ess']}")
