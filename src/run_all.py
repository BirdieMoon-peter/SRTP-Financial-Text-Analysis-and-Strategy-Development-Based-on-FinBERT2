#!/usr/bin/env python3
"""
SRTP: End-to-End Pipeline
==========================
One command to run the complete research pipeline:
  python src/run_all.py

Requires:
  - data/embeddings/  (GPU embeddings - full or 1000-sample)
  - data/csmar_daily_stock.csv (baostock data)
  - data/reports_cleaned.csv
  - data/csmar_index_daily.csv
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def evaluate_factor(factor_values, forward_returns, factor_name, dates=None):
    """Compute IC, RankIC, layered returns for a factor."""
    df = pd.DataFrame({
        "factor": factor_values, "ret": forward_returns,
        "date": pd.to_datetime(dates) if dates is not None else pd.NaT,
    }).dropna()

    if len(df) < 30:
        return None

    # Rank IC
    ic, p_val = stats.spearmanr(df["factor"], df["ret"])
    ic_mean = ic
    ic_t = 0

    if dates is not None and "date" in df.columns:
        ics = []
        for period, group in df.groupby(df["date"].dt.to_period("M")):
            if len(group) >= 10:
                ic_p, _ = stats.spearmanr(group["factor"], group["ret"])
                ics.append(ic_p)
        if ics:
            ic_series = pd.Series(ics)
            ic_mean = ic_series.mean()
            ic_t = ic_mean / ic_series.std() * np.sqrt(len(ic_series)) if ic_series.std() > 0 else 0

    # Layered (5 groups)
    df["group"] = pd.qcut(df["factor"], 5, labels=False, duplicates="drop") + 1
    layer_rets = {}
    for g in range(1, 6):
        g_data = df[df["group"] == g]
        layer_rets[f"G{g}"] = g_data["ret"].mean() if len(g_data) > 0 else np.nan

    ls_ret = layer_rets.get("G5", np.nan) - layer_rets.get("G1", np.nan)
    monotonic = all(
        layer_rets.get(f"G{i}", -np.inf) <= layer_rets.get(f"G{i+1}", np.inf)
        for i in range(1, 5)
    )

    result = {
        "name": factor_name,
        "RankIC": ic_mean,
        "IC_t_stat": ic_t,
        "LS_ret": ls_ret,
        "monotonic": monotonic,
        "n_samples": len(df),
        "G1": layer_rets.get("G1", np.nan),
        "G5": layer_rets.get("G5", np.nan),
    }
    return result


def main():
    log("=" * 60)
    log("SRTP End-to-End Pipeline")
    log("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # 1. Load embeddings
    log("Loading embeddings...")
    embeddings = {}
    for config in ["title", "summary", "full"]:
        path = EMBED_DIR / f"embeddings_{config}.npz"
        if path.exists():
            d = np.load(path)
            embeddings[config] = {
                "last_cls": d["last_cls"],
                "all_cls": d["all_cls"],
                "all_mean": d["all_mean"],
            }
            log(f"  [{config}] {d['last_cls'].shape}")

    # Gap
    gap_path = EMBED_DIR / "embeddings_gap.npz"
    if gap_path.exists():
        embeddings["gap"] = {k: np.load(gap_path)[k] for k in np.load(gap_path).files}

    # Sentiment
    sent_path = EMBED_DIR / "sentiment_finbert.npz"
    if sent_path.exists():
        embeddings["sentiment"] = {k: np.load(sent_path)[k] for k in np.load(sent_path).files}

    n_emb = len(embeddings.get("full", {}).get("last_cls", []))
    log(f"Embedding samples: {n_emb}")

    # 2. Load labels (reports + daily stock data)
    log("\nLoading labels...")
    labels_path = DATA_DIR / "reports_with_labels.csv"
    if labels_path.exists():
        merged = pd.read_csv(labels_path)
        log(f"  Reports with labels: {len(merged):,}")
    else:
        log("  No reports_with_labels.csv, checking stock_labels...")
        stock_labels = DATA_DIR / "stock_labels.csv"
        if stock_labels.exists():
            log(f"  Stock labels available: {pd.read_csv(stock_labels).shape}")
        log("  Using synthetic labels for pipeline test")
        merged = pd.read_csv(DATA_DIR / "reports_cleaned.csv").iloc[:n_emb]
        merged["tradable_date"] = pd.to_datetime(merged["report_date"])
        np.random.seed(42)
        merged["fwd_excess_5d"] = 0.02 * np.random.randn(len(merged))

    # Align embeddings and labels
    if len(merged) > n_emb:
        merged = merged.iloc[:n_emb]
    elif n_emb > len(merged):
        for k in embeddings:
            if isinstance(embeddings[k], dict):
                for kk in embeddings[k]:
                    embeddings[k][kk] = embeddings[k][kk][:len(merged)]

    # 3. Build FHF factors
    log("\nBuilding FHF factors...")
    factors = {}

    # FHF-CLS: Last 4 layers CLS -> PCA1
    if "full" in embeddings:
        cls_4 = embeddings["full"]["all_cls"][:, -4:, :].reshape(n_emb, -1)
        pca = PCA(n_components=1, random_state=42)
        factors["FHF-CLS"] = pca.fit_transform(cls_4).flatten()

        # FHF-LayerMix: All layers CLS -> ElasticNet
        all_cls_flat = embeddings["full"]["all_cls"].reshape(n_emb, -1)
        enet = ElasticNetCV(l1_ratio=[0.5, 0.9], alphas=np.logspace(-4, 0, 10),
                            cv=3, max_iter=3000, random_state=42)
        y = merged.get("fwd_excess_5d", np.random.randn(n_emb)).fillna(0).values
        enet.fit(StandardScaler().fit_transform(all_cls_flat), y)
        factors["FHF-LayerMix"] = enet.predict(
            StandardScaler().fit_transform(all_cls_flat)
        )

        # FHF-TokenPool: Mean pooling all layers -> PCA1
        mean_flat = embeddings["full"]["all_mean"].reshape(n_emb, -1)
        factors["FHF-TokenPool"] = PCA(n_components=1, random_state=42).fit_transform(mean_flat).flatten()

    # FHF-Gap
    if "gap" in embeddings:
        gap_feat = np.column_stack([
            embeddings["gap"]["gap_cls"].mean(axis=1),
            embeddings["gap"]["cos_sim"],
        ])
        factors["FHF-Gap"] = PCA(n_components=1, random_state=42).fit_transform(gap_feat).flatten()

    # Baselines
    if "sentiment" in embeddings:
        probs = embeddings["sentiment"]["probabilities"]
        factors["FinBERT-Sentiment"] = probs[:, 2] - probs[:, 0]  # pos - neg
        factors["FinBERT-PosProb"] = probs[:, 2]

    if "full" in embeddings:
        factors["LastCLS-PCA"] = PCA(n_components=1, random_state=42).fit_transform(
            embeddings["full"]["last_cls"]
        ).flatten()

    # Random baseline
    np.random.seed(42)
    factors["Random"] = np.random.randn(n_emb)

    # Report count baseline
    if "stock_code" in merged.columns:
        counts = merged.groupby("stock_code")["stock_code"].transform("count")
    else:
        counts = np.ones(n_emb)
    factors["ReportCount"] = np.log1p(counts.values)

    log(f"Built {len(factors)} factors: {list(factors.keys())}")

    # 4. Evaluate all factors
    log("\n" + "=" * 60)
    log("Factor Evaluation")
    log("=" * 60)

    target_col = "fwd_excess_5d" if "fwd_excess_5d" in merged.columns else None
    if target_col is None:
        # Try to find any fwd_excess column
        for col in merged.columns:
            if col.startswith("fwd_excess_"):
                target_col = col
                break
    if target_col is None:
        target_col = "fwd_excess_5d"

    returns = merged.get(target_col, pd.Series(np.random.randn(n_emb))).fillna(0).values
    dates = merged.get("tradable_date", pd.date_range("2020-01-01", periods=n_emb))

    results = []
    for name, factor in factors.items():
        factor = np.asarray(factor, dtype=float)
        if len(factor) != len(returns):
            factor = factor[:len(returns)]

        result = evaluate_factor(factor, returns, name, dates)
        if result:
            results.append(result)
            sig = "***" if abs(result["IC_t_stat"]) > 2.58 else (
                "**" if abs(result["IC_t_stat"]) > 1.96 else "")
            log(f"  {name:20s}: RankIC={result['RankIC']:+.4f} "
                f"LS={result['LS_ret']:+.4f} "
                f"Mono={result['monotonic']} {sig}")

    # 5. Save results
    results_df = pd.DataFrame(results).sort_values("RankIC", key=abs, ascending=False)
    results_df.to_csv(RESULTS_DIR / "factor_evaluation.csv", index=False)
    log(f"\nResults saved to {RESULTS_DIR / 'factor_evaluation.csv'}")

    # Top factor
    if len(results_df) > 0:
        best = results_df.iloc[0]
        log(f"\nBest factor: {best['name']} (RankIC={best['RankIC']:.4f})")

    # Save full results as JSON
    with open(RESULTS_DIR / "pipeline_results.json", "w") as f:
        json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in r.items()} for r in results], f, indent=2)

    log("\nPipeline complete!")


if __name__ == "__main__":
    main()
