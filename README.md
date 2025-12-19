# 年报与情感分析流水线

聚合了从巨潮资讯抓取年报链接、批量下载 PDF、清洗分句并用 FinBERT 生成文本向量的脚本，同时附带雪球评论、财经新闻和股票行情等辅助抓取工具。

- 年报抓取下载：`年报/年报链接爬取.py`、`年报/年报下载.py`
- 文本清洗与分句：`年报分析/process_reports.py`
- 向量化：`年报分析/analyze_sentiment.py`
- 其他数据：`财经新闻.py`、`雪球评论.py`、`股票数据/获取指定代码股票数据.py`

## 目录结构
```
.
├── README.md
├── .gitignore
├── 年报/                     # 年报链接 Excel 与爬取、下载脚本
│   ├── 年报链接爬取.py
│   ├── 年报下载.py
│   └── 年报链接_*.xlsx
├── PDF文件/                  # 下载的年报 PDF（按行业/年份分类，已忽略）
├── 年报分析/                 # 文本处理与向量化
│   ├── process_reports.py    # PDF 清洗、分句并写 CSV
│   ├── analyze_sentiment.py  # FinBERT 生成文档向量
│   ├── csv_output/           # 分句后 CSV
│   ├── finbert_embeddings.csv
│   └── stopwords_full.txt
├── 股票数据/
│   └── 获取指定代码股票数据.py
├── 财经新闻.py
├── 雪球评论.py
├── FinBERT-main/             # 上游 FinBERT 参考
└── 研究日志/                 # 研究记录（已忽略）
```

## 环境依赖
- Python 3.9+，建议创建虚拟环境。
- 主要库：`pandas`、`requests`、`pdfplumber`、`transformers`、`torch`、`tqdm`、`openpyxl`、`lxml`、`beautifulsoup4`、`tushare`。
- Hugging Face 模型：`valuesimplex-ai-lab/FinBERT2-base`（首次运行会自动下载，需要网络）。
- 可选：配置 Tushare token（行情抓取脚本内有占位）。

安装示例：
```bash
pip install pandas requests pdfplumber transformers torch tqdm openpyxl lxml beautifulsoup4 tushare
```

## 使用流程
1) **爬取年报链接**  
   在 `年报/年报链接爬取.py` 中调整 `setYear` 或批量年份范围，运行生成 `年报链接_YYYY.xlsx`。

2) **批量下载年报 PDF**  
   将上一步生成的 Excel 放在 `年报/`，运行 `python 年报/年报下载.py`。PDF 会保存到 `PDF文件/<企业类别>/<年份>年年报/`。

3) **分句抽取**  
   把 PDF 目录复制/移动到 `年报分析/PDF文件/`（脚本按此路径查找），运行：  
   ```bash
   python 年报分析/process_reports.py
   ```  
   输出：`年报分析/csv_output/2018_000001_平安银行.csv` 等（包含分句）。

4) **生成 FinBERT 向量**  
   ```bash
   python 年报分析/analyze_sentiment.py
   ```  
   输出：`年报分析/finbert_embeddings.csv`，包含 `year/company_code/company_name` 及向量列 `vec_0...`。

5) **（可选）行情与其他数据**  
   - `股票数据/获取指定代码股票数据.py`：根据 `PDF文件/` 中的股票代码抓取历史收盘价，需在脚本内替换 Tushare token。输出 `补.xlsx`。  
   - `财经新闻.py`、`雪球评论.py`：示例爬虫，需根据目标股票或 Cookie 适当修改。

## 文件命名与约定
- 年报 PDF 文件名建议：`<公司代码>_<公司简称>_<年份>.pdf`，脚本据此解析元数据。
- 处理链默认中文内容，CSV 使用 `utf-8-sig` 编码便于 Excel 打开。

## 常见提示
- Hugging Face 模型与 Tushare 调用需要联网；若在离线环境请预先下载模型或缓存数据。
- `process_reports.py` 会跳过已存在的 CSV，更新 PDF 后请删除对应 CSV 再运行。
- `.gitignore` 已忽略大体量 PDF 与研究日志，必要时自行管理备份。
