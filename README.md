# SRTP: FinBERT 情绪信号增强与隐层互补表征

浙江大学大学生创新创业训练计划（SRTP）项目：

**基于 FinBERT 情绪信号增强与隐层互补表征的 A 股分析师研报文本因子研究**

本项目研究中文 A 股分析师研报中的文本信息是否能够预测未来短期横截面收益。项目的最终结论不是“隐层表征显著优于情绪输出”，而是：

1. FinBERT 情绪输出本身具有弱但方向稳定的收益预测力。
2. 隐层表征单独不强，但能提供与情绪输出互补的语义信息。
3. 主要提升来自金融因子工程后处理，包括行业中性化、分析师共识聚合和覆盖度缩放。
4. 增强后的文本因子存在一定经济价值，但对交易成本和样本外稳定性敏感。

完整论文见 [thesis/thesis.pdf](thesis/thesis.pdf)。

## 核心结果

样本覆盖 2020--2026 年 A 股分析师研报，共 98,699 条原始研报，匹配后得到 76,742 条有效样本，涉及 2,820 只股票和 73 个月度截面。

| 因子/方法 | Rank IC | t 值 | 解释 |
|---|---:|---:|---|
| FinBERT-Sentiment | 0.0154 | 1.72* | 原始情绪输出，具有弱预测力 |
| FinBERT-Sentiment-Neut | 0.0212 | 2.24** | 行业中性化后信号增强 |
| FHF-CLS-PLS | 0.0131 | 1.58 | 隐层单因子未超越情绪输出 |
| FHF-Ensemble | 0.0242 | 2.26** | 多视角隐层表征存在互补价值 |
| FHF-Gap-PLS | -0.0112 | -1.57 | 标题-摘要语义差异呈负向预测 |
| CovScaled | **0.0265** | **2.72*** | 行业中性化 + 共识聚合 + 覆盖度缩放，表现最优 |

注：`* p<0.10`，`** p<0.05`，`*** p<0.01`。结果以论文主实验设定为准。

## 方法框架

### 1. 文本信号

- `FinBERT-Sentiment`：FinBERT 最终情绪输出，定义为正面概率减负面概率。
- `FHF-CLS`：后四层 `[CLS]` 向量拼接后降维。
- `FHF-LayerMix`：全部 12 层 `[CLS]` 向量拼接后降维。
- `FHF-TokenPool`：token mean pooling 表征。
- `FHF-Gap`：标题与摘要之间的语义差异。
- `FHF-Ensemble`：多类隐层因子的组合信号。

### 2. 金融因子工程后处理

最终结果表明，后处理比单纯增加模型复杂度更关键：

- 行业中性化：剥离行业配置噪声。
- 分析师共识聚合：同一股票同一交易日多篇研报先聚合。
- 覆盖度缩放：多分析师覆盖的信号权重更高。
- 时间衰减加权：近期研报赋予更高权重。

最优信号 `CovScaled` 不依赖隐层嵌入，仅基于 FinBERT 情绪输出、行业分类和研报覆盖数量构造。

### 3. 实证检验

项目使用以下方式评估文本因子：

- 月度 Rank IC / ICIR
- 十分组分层收益
- Fama-MacBeth 横截面回归
- 行业、市值、动量、流动性正交化
- 不同预测窗口和市场阶段稳健性检验
- placebo 实验
- 多重检验调整
- 交易成本敏感性回测

## 策略回测结论

`CovScaled` 因子在月频调仓、Top-20% 等权多头组合下：

| 成本假设 | 年化收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| 零成本 | 3.36% | 0.84 | -5.12% |
| 0.05% | 2.78% | 0.69 | -5.67% |
| 0.10% | 2.21% | 0.55 | -6.23% |
| 0.30% | -0.09% | -0.02 | -7.95% |

因此，文本因子更适合作为多因子框架中的辅助 alpha 信号，而不是独立高换手交易策略。

## 项目结构

```text
SRTP/
├── src/                         # 源代码
│   ├── preprocess.py            # 文本预处理
│   ├── download_csmar.py        # CSMAR 数据下载
│   ├── process_csmar.py         # CSMAR 数据清洗
│   ├── download_baostock.py     # Baostock 数据下载
│   ├── download_models.py       # 本地模型下载
│   ├── extract_embeddings.py    # FinBERT 嵌入与情绪输出提取
│   ├── extract_gpu.py           # GPU 嵌入提取
│   ├── extract_3090.py          # RTX 3090 优化提取
│   ├── build_labels.py          # 收益标签构建
│   ├── build_labels_full.py     # 完整标签构建
│   ├── fhf_factors.py           # FHF 因子构造
│   ├── signal_enhancement.py    # 行业中性化、共识聚合、覆盖度缩放
│   ├── supplementary_analysis.py # 稳健性与补充分析
│   ├── backtest.py              # 基础回测
│   ├── backtest_improved.py     # CovScaled 等增强信号回测
│   ├── generate_tables.py       # 论文表格生成
│   ├── analyze.py               # 综合分析与图表
│   ├── run_pipeline.py          # 分步流水线
│   └── run_all.py               # 一键端到端流水线
├── data/                        # 数据目录
│   ├── embeddings/              # 预提取嵌入向量
│   ├── reports_cleaned.csv      # 清洗后研报
│   ├── reports_with_labels.csv  # 研报与收益标签匹配结果
│   └── csmar_*.csv              # 市场、行业、停牌、ST 等数据
├── models/                      # 本地模型文件
├── results/                     # 实证结果与回测输出
├── logs/                        # 运行日志
├── thesis/                      # 论文 LaTeX 源文件与 PDF
├── 项目申报文档/                # SRTP 申报和结题材料
└── 相关论文/                    # 参考文献 PDF
```

## 环境依赖

建议环境：

- Python 3.8+
- PyTorch
- Transformers
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- LightGBM（可选）
- Baostock（若需重新下载行情数据）

## 快速开始

### 1. 数据准备

```bash
python src/download_csmar.py
python src/process_csmar.py
python src/download_baostock.py
python src/preprocess.py
```

### 2. 下载模型

```bash
python src/download_models.py
```

### 3. 提取 FinBERT 输出和隐层嵌入

```bash
python src/extract_3090.py
```

也可使用通用版本：

```bash
python src/extract_gpu.py
```

### 4. 构建收益标签

```bash
python src/build_labels_full.py
```

### 5. 构建因子与运行实验

```bash
python src/run_all.py
```

根据具体实验，也可以单独运行：

```bash
python src/signal_enhancement.py
python src/supplementary_analysis.py
python src/backtest_improved.py
python src/generate_tables.py
```

## 主要输出

- [thesis/thesis.pdf](thesis/thesis.pdf)：完整研究论文。
- `results/final_results.json`：主要因子结果。
- `results/signal_enhancement_results.json`：信号增强结果。
- `results/backtest_results.json`：组合回测结果。
- `results/robustness_windows.json`：不同预测窗口稳健性。
- `results/robustness_periods.json`：不同市场阶段稳健性。
- `results/orthogonalization.json`：因子正交化检验。
- `results/layer_importance.csv`：逐层分析。
- `thesis/figures/`：论文图表。

## 研究定位

本项目不主张文本因子可以直接构成稳定可交易的独立策略。更稳妥的定位是：

- 文本信号真实存在，但属于低信噪比弱信号。
- FinBERT 情绪输出经合理后处理后比复杂隐层单因子更有效。
- 隐层表征的价值在于补充语义信息，适合作为组合中的互补信号。
- 实际应用必须考虑交易成本、容量、换手率和样本外稳定性。

