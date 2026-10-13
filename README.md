# 植物光竞争进化博弈模型 —— 快速上手指南

> **Code & data for:** Hang Xu, *Light competition as an evolutionary game identifies partial shade-avoidance attenuation as a target for high-density breeding* (Modelling/Theory; submitted to *New Phytologist*).
> **Author:** Hang Xu (许航), China Agricultural University, Beijing, China — ORCID [0009-0009-3844-4524](https://orcid.org/0009-0009-3844-4524). Single author; no competing interests; no specific funding.
> **License:** source code **MIT**; data and figures **CC-BY 4.0**.
> **Reproducibility:** `python src/main.py` regenerates every figure (Fig. 1–7, Fig. S1–S4 in the manuscript) and all 19 CSVs; Python 3, Monte Carlo n = 6,000, random seed 42 (see `requirements.txt`).

---

## 30秒跑通项目

```bash
# 1. 进入项目目录
cd light_competition_game

# 2. 安装依赖（建议用虚拟环境）
pip install -r requirements.txt

# 3. 运行全部分析（生成所有图表+CSV）
cd src
python main.py
```

运行成功后：
- `figures/` 目录下会出现 Fig1–Fig10 共10张主图 + FigS 文献对标图（共11张）
- `data/` 目录下会出现19个CSV（核心/育种/演化/光信号 + 4个文献对标 + 2个分子耦合
  + 5个Burbano田间实证 + 4个敏感性分析）
- 控制台会打印4项文献对标、分子跨层一致性、Burbano反演与全局稳健性结果

也可单独运行验证模块：

```bash
cd src
python benchmark.py             # 4项文献对标（Gersani/Bongers/Burbano/公地悲剧）
python molecular.py             # 分子-表型-群体跨层耦合
python empirical_validation.py  # Burbano2025田间数据反演（g 与悲剧损失 L）
python sensitivity.py           # 全局Monte Carlo敏感性与稳健性（Fig10）
```

## 项目结构

```
light_competition_game/
├── src/
│   ├── config.py                # 【常改】所有参数在这里调
│   ├── model.py                 # 【核心】博弈模型、ESS解析解
│   ├── simulation.py            # 【核心】演化模拟(IBM/PIP)验证
│   ├── benchmark.py             # 【验证】4项已发表结论的文献对标
│   ├── molecular.py             # 【跨层】phyB/CRY/PIF/auxin→策略s→产量
│   ├── empirical_validation.py  # 【实证】Burbano田间产量点反演 g、L（Fig9）
│   ├── sensitivity.py           # 【稳健】OAT+全局Monte Carlo敏感性（Fig10）
│   ├── visualization.py         # 主图生成
│   └── main.py                  # 一键运行入口（已集成上述全部模块）
├── figures/                     # Fig1–Fig10 + FigS 文献对标图
├── data/                        # 19个数值结果CSV（含数字化原始数据）
├── docs/                        # 00研究方案 01数据方案 02精读模板 03/04文献精读 05 Intro大纲 06实证反演 07敏感性 08 Results 09 Methods 10 Discussion 11 Introduction 12 Abstract/Title/Keywords英文初稿(均含中文导读)
└── papers/                      # 7篇核心文献PDF
```

**四层验证逻辑**
- `benchmark.py`：新模型先复现独立发表的经典定性结论（Gersani2001 根系公地悲剧、
  Bongers2019 密度依赖cESS、Burbano2025 弱敏感株系HD增产、ESS≠群体最优），全部 PASS。
- `molecular.py`：以野生型 ESS 为演化锚点，分子响应增益 g 决定基因型把避荫可塑性
  缩放多少（WT g=1 精确实现ESS，误差为0；IL2-2/pif457/sav3 等 g<1 向群体最优回落）。
- `empirical_validation.py`：用 Burbano2025 番茄田间独立数据反演。尺度无关桥梁
  G=(1−g²)·L/(1−L)；HD-Spain 实测显著增产13.5%，反演番茄HD野生型悲剧损失
  L≈12–16%、IL2-2 增益 g≈0.35–0.55（Fig9）。
- `sensitivity.py`：6000 组参数全局扫描，四条结构型结论（悲剧存在、密度单调、
  HD>LD、s_opt>0）在 3974 个内点样本中 **100% 稳健**；损失量级主要由竞争参数
  k1、β 决定（N=800 的 5–95% 分位为 0.1–15.8%），方向靠模型、量级靠数据（Fig10）。

## 新手第一步：动手改参数，理解模型

打开 `src/config.py` 或 `src/model.py` 末尾的测试代码，尝试：

```python
# 在 src/ 目录下启动 python
python
>>> from model import GameParams, tragedy_gap
>>> p = GameParams()
>>> ess, s_opt, gap = tragedy_gap(500, p)
>>> print(f"密度500: ESS={ess:.3f}, 群体最优={s_opt:.3f}, 偏差={gap:.3f}")

# 改参数试试：成本系数c翻倍后会怎样？
>>> p2 = GameParams(c=0.6)
>>> ess2, s_opt2, gap2 = tragedy_gap(500, p2)
>>> print(f"成本翻倍: ESS={ess2:.3f}, 群体最优={s_opt2:.3f}, 偏差={gap2:.3f}")
```

**核心公式（必须能自己推导）**：
- ESS:   s* = [(k₀γ + k₁N^β) / (cα)]^(1/(α-1))
- 群体最优: s_opt = [k₀γ / (cα)]^(1/(α-1))
- 偏差:   Δs = s* - s_opt > 0（公地悲剧）

## 常见问题

**Q: 中文显示成方块怎么办？**
A: 图表中的中文需要系统安装中文字体。Linux执行 `sudo apt install fonts-noto-cjk`，或把visualization.py中的中文标题改成英文。

**Q: 演化模拟跑得慢？**
A: 修改config.py中 `population_size`（默认1000）和 `generations`（默认500），调试时可改为200/100。

**Q: 想只重新生成图表，不跑模拟？**
A: `python -c "from visualization import generate_all_figures; from model import GameParams; generate_all_figures(GameParams(), '../figures')"`
