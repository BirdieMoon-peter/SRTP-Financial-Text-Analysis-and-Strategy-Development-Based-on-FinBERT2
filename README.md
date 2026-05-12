# SRTP: FinBERT 隐层文本因子研究

基于 FinBERT 隐藏层嵌入的股票选择因子构建与实证研究。浙江大学大学生创新创业训练计划（SRTP）项目。

## 概述

传统的文本因子方法仅使用 FinBERT 的最终输出层（[CLS] token 或情感概率），忽略了中间层包含的丰富语义信息。本项目从 FinBERT 的 12 层 Transformer 隐藏层中提取嵌入表征，通过有监督降维构造一类新的 **FinBERT 隐层因子（FinBERT Hidden Factors, FHF）**，并在 A 股市场进行实证检验。

### 核心方法

| 因子 | 方法 | 说明 |
|------|------|------|
| FHF-CLS | 后4层 [CLS] → PCA | 聚合深层语义信息 |
| FHF-LayerMix | 全层 [CLS] → ElasticNet | 有监督层间加权 |
| FHF-TokenPool | 全层 Mean Pooling → PCA | 聚合 token 级信息 |
| FHF-Gap | 标题-正文嵌入差异 | 捕捉信息增量 |
| FinBERT-Sentiment | 输出层情感概率 | 传统基线 |
| LastCLS-PCA | 末层 [CLS] → PCA | 消融基线 |

## 项目结构

```
SRTP/
├── src/                        # 源代码
│   ├── preprocess.py           # 文本预处理
│   ├── download_csmar.py       # CSMAR 数据库下载
│   ├── process_csmar.py        # CSMAR 数据清洗
│   ├── download_baostock.py    # Baostock 数据下载
│   ├── download_models.py      # 模型下载（FinBERT 等）
│   ├── extract_embeddings.py   # 嵌入提取（通用）
│   ├── extract_gpu.py          # 嵌入提取（GPU 加速）
│   ├── extract_3090.py         # 嵌入提取（RTX 3090 优化）
│   ├── build_labels.py         # 标签构建
│   ├── build_labels_full.py    # 完整标签构建
│   ├── fhf_factors.py          # FHF 因子构造（核心算法）
│   ├── backtest.py             # 回测与实证检验
│   ├── analyze.py              # 综合分析 & 论文图表
│   ├── interpretability.py     # 可解释性分析
│   ├── signal_enhancement.py   # 信号增强
│   ├── supplementary_analysis.py # 补充分析
│   ├── generate_tables.py      # 论文表格生成
│   ├── complete_all_gaps.py    # 缺口数据补全
│   ├── run_pipeline.py         # 分步流水线
│   └── run_all.py              # 一键端到端流水线
├── data/                       # 数据目录
│   ├── embeddings/             # 预提取的嵌入向量
│   ├── reports_cleaned.csv     # 清洗后的分析师报告
│   ├── reports_with_labels.csv # 带标签的报告
│   └── csmar_*.csv             # CSMAR 市场数据
├── models/                     # 本地模型文件
├── results/                    # 实证结果输出
├── thesis/                     # 论文 LaTeX 源文件
│   ├── thesis.tex              # 论文主文件
│   ├── thesis.pdf              # 编译后的论文
│   ├── figures/                # 论文图表
│   └── ref.bib                 # 参考文献
└── logs/                       # 运行日志
```

## 环境依赖

- Python 3.8+
- PyTorch ≥ 1.10
- Transformers ≥ 4.20
- scikit-learn
- pandas, numpy, scipy
- matplotlib
- LightGBM（可选，用于 FHF-LayerMix-LGB 方法）

## 快速开始

### 1. 数据准备

```bash
# 下载分析师报告（CSMAR 数据库）
python src/download_csmar.py

# 文本预处理
python src/preprocess.py

# 下载模型
python src/download_models.py
```

### 2. 提取 FinBERT 嵌入

```bash
# GPU 环境提取嵌入
python src/extract_gpu.py

# 或使用 RTX 3090 优化版
python src/extract_3090.py
```

### 3. 构建标签

```bash
python src/build_labels_full.py
```

### 4. 运行端到端流水线

```bash
# 一键运行：因子构建 + 评估 + 图表生成
python src/run_all.py
```

## 评估指标

- **Rank IC / ICIR**：因子值与未来收益的秩相关系数
- **分层回测**：按因子值分5组，检验多空收益单调性
- **Fama-MacBeth 回归**：控制行业、市值后的因子溢价
- **因子单调性**：分组收益是否严格递增

## 实验结果摘要

基于 29,395 份中文分析师报告（2017—2024），在 A 股市场进行实证检验：

- **FHF-LayerMix** 实现日度 RankIC = 0.0321（ICIR = 0.89），优于传统 FinBERT 情感信号
- 隐层因子在控制行业和市值后仍保持显著的超额收益预测能力
- 标题-正文嵌入差异（FHF-Gap）蕴含增量信息，标题因子在短期（5日）表现突出

详细结果见 `thesis/thesis.pdf`。

## 项目产出

- `results/factor_evaluation.csv` — 全部因子评估结果
- `thesis/figures/` — 论文图表（因子对比、分层收益、IC 热力图等）
- `thesis/thesis.pdf` — 完整论文
