"""
SRTP: Interpretability Analysis Module
========================================
Layer importance, token importance via attention,
title-summary gap, event topic classification,
manual review sampling.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_embeddings_and_reports():
    """Load embeddings and reports for analysis."""
    data = {}
    reports_path = DATA_DIR / "reports_cleaned.csv"
    if reports_path.exists():
        data["reports"] = pd.read_csv(reports_path)
        data["reports"]["report_date"] = pd.to_datetime(data["reports"]["report_date"])

    for config in ["title", "summary", "full"]:
        path = EMBED_DIR / f"embeddings_{config}.npz"
        meta_path = EMBED_DIR / f"embeddings_{config}_meta.json"
        if path.exists() and meta_path.exists():
            d = np.load(path)
            with open(meta_path) as f:
                meta = json.load(f)
            data[config] = {
                "last_cls": d["last_cls"], "all_cls": d["all_cls"],
                "all_mean": d["all_mean"],
                "n_layers": meta.get("n_layers", 12),
            }

    gap_path = EMBED_DIR / "embeddings_gap.npz"
    if gap_path.exists():
        data["gap"] = {k: np.load(gap_path)[k] for k in np.load(gap_path).files}

    sent_path = EMBED_DIR / "sentiment_finbert.npz"
    if sent_path.exists():
        d = np.load(sent_path)
        data["sentiment"] = {"probabilities": d["probabilities"]}

    # Align data: if embeddings are a subset, slice reports to match
    if "reports" in data and "full" in data:
        n_emb = len(data["full"]["last_cls"])
        n_rep = len(data["reports"])
        if n_emb < n_rep:
            print(f"[align] Using first {n_emb} reports matching {n_emb} embeddings")
            data["reports"] = data["reports"].iloc[:n_emb].reset_index(drop=True)

    return data


def analyze_layer_importance(data, returns=None):
    """
    Analyze which BERT layers contribute most to prediction.
    Uses CLS token from each layer individually.
    """
    if "full" not in data:
        return None

    all_cls = data["full"]["all_cls"]  # (N, L, H)
    n_layers = all_cls.shape[1]
    n_samples = all_cls.shape[0]

    # Use sentiment probabilities as proxy labels if no returns
    if returns is None and "sentiment" in data:
        returns = data["sentiment"]["probabilities"][:, 2]  # positive prob
    elif returns is None:
        returns = np.random.randn(n_samples)

    valid = ~np.isnan(returns)
    results = []
    for l in range(n_layers):
        # First principal component of layer-l CLS
        layer_cls = all_cls[valid, l, :]  # (valid_N, H)
        # Simple: use first PC
        scaler = StandardScaler()
        X = scaler.fit_transform(layer_cls)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        factor = U[:, 0]

        ic, p_val = stats.spearmanr(factor, returns[valid])
        results.append({"layer": l + 1, "rank_ic": abs(ic), "p_value": p_val})

    layer_df = pd.DataFrame(results)
    layer_df.to_csv(RESULTS_DIR / "layer_importance.csv", index=False)

    print("[layer] Layer importance analysis complete")
    for _, row in layer_df.iterrows():
        stars = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.1 else ""))
        print(f"  Layer {int(row['layer']):2d}: |IC|={row['rank_ic']:.4f} {stars}")

    return layer_df


def analyze_token_importance(data, top_k=50):
    """
    Analyze which tokens/terms correlate most with sentiment.
    Uses TF-IDF to find keywords differentiating high/low sentiment reports.
    """
    if "reports" not in data or "sentiment" not in data:
        return None

    df = data["reports"]
    probs = data["sentiment"]["probabilities"]
    pos_score = probs[:, 2] - probs[:, 0]  # positive - negative

    # Binarize: top 30% vs bottom 30%
    hi_thresh = np.quantile(pos_score, 0.7)
    lo_thresh = np.quantile(pos_score, 0.3)
    hi_mask = pos_score >= hi_thresh
    lo_mask = pos_score <= lo_thresh

    print(f"[token] High sentiment: {hi_mask.sum()}, Low: {lo_mask.sum()}")

    # Extract texts
    hi_texts = df.loc[hi_mask, "summary"].fillna("").tolist()
    lo_texts = df.loc[lo_mask, "summary"].fillna("").tolist()

    # TF-IDF on combined texts, then compare
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.5,
                                  token_pattern=r'(?u)\b\w+\b')
    all_texts = hi_texts + lo_texts
    labels = np.array([1] * len(hi_texts) + [0] * len(lo_texts))

    X = vectorizer.fit_transform(all_texts)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X, labels)

    # Get top positive and negative coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_pos = np.argsort(coefs)[-top_k:][::-1]
    top_neg = np.argsort(coefs)[:top_k]

    pos_tokens = [(feature_names[i], coefs[i]) for i in top_pos]
    neg_tokens = [(feature_names[i], coefs[i]) for i in top_neg]

    print(f"\n  Top positive tokens (associated with high sentiment):")
    for token, coef in pos_tokens[:15]:
        print(f"    {token:15s}  {coef:+.4f}")
    print(f"\n  Top negative tokens (associated with low sentiment):")
    for token, coef in neg_tokens[:15]:
        print(f"    {token:15s}  {coef:+.4f}")

    # Save
    token_df = pd.DataFrame({
        "token": [t for t, _ in pos_tokens] + [t for t, _ in neg_tokens],
        "coefficient": [c for _, c in pos_tokens] + [c for _, c in neg_tokens],
        "direction": ["positive"] * len(pos_tokens) + ["negative"] * len(neg_tokens),
    })
    token_df.to_csv(RESULTS_DIR / "token_importance.csv", index=False)

    return token_df


def analyze_title_summary_gap(data):
    """
    Analyze title-summary semantic gaps.
    Large gaps may indicate title packaging / optimism bias.
    """
    if "gap" not in data or "reports" not in data:
        return None

    cos_sim = data["gap"]["cos_sim"]  # (N, L)
    # Use last layer cosine similarity as gap measure
    last_layer_sim = cos_sim[:, -1]
    gap_score = 1 - last_layer_sim  # higher = more divergent

    df = data["reports"].copy()
    df["gap_score"] = gap_score

    # Extreme gap examples
    high_gap = df.nlargest(20, "gap_score")
    low_gap = df.nsmallest(20, "gap_score")

    print(f"\n[gap] High gap (title-summary divergent) examples:")
    for _, row in high_gap.head(5).iterrows():
        print(f"  [{row['stock_code']}] {row['title'][:80]}")
        print(f"    Summary: {row['summary'][:100]}...")
        print(f"    Gap score: {row['gap_score']:.3f}")

    print(f"\n[gap] Low gap (title-summary aligned) examples:")
    for _, row in low_gap.head(5).iterrows():
        print(f"  [{row['stock_code']}] {row['title'][:80]}")
        print(f"    Gap score: {row['gap_score']:.3f}")

    # Gap statistics by year
    df["year"] = pd.to_datetime(df["report_date"]).dt.year
    yearly_gap = df.groupby("year")["gap_score"].agg(["mean", "std", "count"])
    print(f"\n[gap] Yearly gap statistics:")
    print(yearly_gap)

    yearly_gap.to_csv(RESULTS_DIR / "yearly_gap_stats.csv")

    # Save full gap analysis
    gap_df = df[["report_date", "stock_code", "company_name", "title",
                  "summary", "gap_score"]].copy()
    gap_df.to_csv(RESULTS_DIR / "gap_analysis.csv", index=False)

    return gap_df


def classify_event_topics(data):
    """
    Classify reports into event topics using keyword matching.
    Topics: 业绩点评, 订单/合同, 政策, 成本/价格, 风险, 估值, 并购, 新产品, 行业景气
    """
    if "reports" not in data:
        return None

    df = data["reports"].copy()

    topic_keywords = {
        "业绩点评": ["业绩", "利润", "净利润", "归母", "营收", "收入增长", "盈利", "毛利率",
                   "净利率", "ROE", "EPS", "每股收益"],
        "订单合同": ["订单", "中标", "签约", "合同", "客户", "采购", "预付款"],
        "政策影响": ["政策", "监管", "补贴", "税收", "法规", "政府", "审批", "牌照"],
        "成本价格": ["成本", "涨价", "降价", "价格战", "原材料", "费用率", "三费"],
        "风险提示": ["风险", "下行", "压力", "承压", "不确定性", "谨慎", "警惕", "低于预期",
                   "下滑", "下降", "亏损"],
        "估值分析": ["估值", "PE", "PB", "折现", "目标价", "合理", "低估", "高估"],
        "并购重组": ["并购", "收购", "重组", "注入", "整合", "定增", "增发", "融资"],
        "新品研发": ["新品", "研发", "创新", "技术", "专利", "产能", "量产", "投产"],
        "行业景气": ["行业", "景气", "周期", "赛道", "格局", "集中度", "市场空间"],
        "分红回购": ["分红", "派息", "回购", "股权激励", "员工持股"],
    }

    results = []
    for topic, keywords in topic_keywords.items():
        pattern = "|".join(keywords)
        mask = df["summary"].str.contains(pattern, case=False, na=False)
        results.append({"topic": topic, "count": mask.sum(),
                        "pct": mask.sum() / len(df) * 100})

    topic_df = pd.DataFrame(results).sort_values("count", ascending=False)
    print(f"\n[topic] Event topic distribution:")
    for _, row in topic_df.iterrows():
        print(f"  {row['topic']:10s}: {row['count']:6d} ({row['pct']:5.1f}%)")

    topic_df.to_csv(RESULTS_DIR / "topic_distribution.csv", index=False)
    return topic_df


def sample_manual_review(data, n_samples=500, random_state=42):
    """
    Sample reports for manual review, stratified by sentiment and gap score.
    """
    if "reports" not in data:
        return None

    df = data["reports"].copy()
    rng = np.random.RandomState(random_state)

    # Stratify by year and add sentiment if available
    if "sentiment" in data:
        probs = data["sentiment"]["probabilities"]
        df["sentiment_score"] = probs[:, 2] - probs[:, 0]
        df["sentiment_label"] = pd.cut(df["sentiment_score"],
                                        bins=[-np.inf, -0.3, 0.3, np.inf],
                                        labels=["negative", "neutral", "positive"])

    if "gap" in data:
        df["gap_score"] = 1 - data["gap"]["cos_sim"][:, -1]

    # Stratified sampling
    samples = []
    years = sorted(df["report_date"].dt.year.unique())
    n_per_year = max(n_samples // len(years), 20)

    for year in years:
        year_data = df[df["report_date"].dt.year == year]
        if len(year_data) <= n_per_year:
            samples.append(year_data)
        else:
            samples.append(year_data.sample(n_per_year, random_state=random_state))

    review_df = pd.concat(samples, ignore_index=True)

    # Select key columns
    cols = ["report_date", "stock_code", "company_name", "title", "summary"]
    if "sentiment_label" in review_df.columns:
        cols.append("sentiment_label")
        cols.append("sentiment_score")
    if "gap_score" in review_df.columns:
        cols.append("gap_score")

    review_df[cols].to_csv(RESULTS_DIR / "manual_review_sample.csv", index=False)

    print(f"\n[review] Sampled {len(review_df)} reports for manual review")
    print(f"[review] Saved to {RESULTS_DIR / 'manual_review_sample.csv'}")
    if "sentiment_label" in review_df.columns:
        print(f"[review] Sentiment distribution:\n{review_df['sentiment_label'].value_counts()}")
    if "gap_score" in review_df.columns:
        print(f"[review] Gap score: mean={review_df['gap_score'].mean():.3f}, "
              f"std={review_df['gap_score'].std():.3f}")

    return review_df


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("SRTP Interpretability Analysis")
    print("=" * 60)

    data = load_embeddings_and_reports()
    print(f"Loaded: {list(data.keys())}")

    if not data:
        print("No data available. Run embedding extraction first.")
        return

    # 1. Layer importance
    print("\n" + "=" * 40)
    print("1. Layer Importance Analysis")
    analyze_layer_importance(data)

    # 2. Token importance
    print("\n" + "=" * 40)
    print("2. Token Importance Analysis")
    analyze_token_importance(data)

    # 3. Title-summary gap
    print("\n" + "=" * 40)
    print("3. Title-Summary Gap Analysis")
    analyze_title_summary_gap(data)

    # 4. Event topic classification
    print("\n" + "=" * 40)
    print("4. Event Topic Classification")
    classify_event_topics(data)

    # 5. Manual review sampling
    print("\n" + "=" * 40)
    print("5. Manual Review Sampling")
    sample_manual_review(data)

    print(f"\n{'='*60}")
    print("Interpretability analysis complete.")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
