"""
SRTP: Comprehensive Analysis and Plotting
==========================================
Loads embeddings, builds FHF factors, runs backtests,
and generates all figures/tables for the thesis.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "thesis" / "figures"

# Plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_all_data():
    """Load embeddings, reports, and auxiliary data."""
    data = {}

    # Embeddings
    for config in ["title", "summary", "full"]:
        path = EMBED_DIR / f"embeddings_{config}.npz"
        meta_path = EMBED_DIR / f"embeddings_{config}_meta.json"
        if path.exists() and meta_path.exists():
            d = np.load(path)
            with open(meta_path) as f:
                meta = json.load(f)
            data[config] = {
                "last_cls": d["last_cls"],
                "all_cls": d["all_cls"],
                "all_mean": d["all_mean"],
                "n_layers": meta.get("n_layers", 12),
                "hidden_dim": meta.get("hidden_dim", 768),
            }

    # Gap
    gap_path = EMBED_DIR / "embeddings_gap.npz"
    if gap_path.exists():
        d = np.load(gap_path)
        data["gap"] = {k: d[k] for k in d.files}

    # Sentiment
    sent_path = EMBED_DIR / "sentiment_finbert.npz"
    if sent_path.exists():
        d = np.load(sent_path)
        data["sentiment"] = {k: d[k] for k in d.files}

    # Reports
    reports_path = DATA_DIR / "reports_cleaned.csv"
    if reports_path.exists():
        data["reports"] = pd.read_csv(reports_path)
        data["reports"]["report_date"] = pd.to_datetime(data["reports"]["report_date"])

    # Index data
    idx_path = DATA_DIR / "index_closes.csv"
    if idx_path.exists():
        data["index"] = pd.read_csv(idx_path, index_col=0, parse_dates=True)

    # Industry
    ind_path = DATA_DIR / "industry_mapping.csv"
    if ind_path.exists():
        data["industry"] = pd.read_csv(ind_path, dtype={"stock_code": str})

    return data


def build_factor_features(data):
    """Build comprehensive feature set from embeddings."""
    features = {}

    # FHF-CLS: last 4 layers CLS
    if "full" in data:
        all_cls = data["full"]["all_cls"]  # (N, L, H)
        L = all_cls.shape[1]
        features["FHF_CLS"] = all_cls[:, -4:, :].reshape(all_cls.shape[0], -1)

    # FHF-LayerMix: all layers CLS
    if "full" in data:
        all_cls = data["full"]["all_cls"]
        features["FHF_LayerMix"] = all_cls.reshape(all_cls.shape[0], -1)

    # FHF-TokenPool: mean pooling all layers
    if "full" in data:
        all_mean = data["full"]["all_mean"]
        features["FHF_TokenPool"] = all_mean.reshape(all_mean.shape[0], -1)

    # FHF-Gap: title-summary difference
    if "gap" in data:
        gap_features = []
        if "gap_cls" in data["gap"]:
            gap_features.append(data["gap"]["gap_cls"].mean(axis=1))
        if "cos_sim" in data["gap"]:
            gap_features.append(data["gap"]["cos_sim"])
        if gap_features:
            features["FHF_Gap"] = np.concatenate(gap_features, axis=1)

    # Baseline: FinBERT final sentiment
    if "sentiment" in data and "probabilities" in data["sentiment"]:
        probs = data["sentiment"]["probabilities"]  # (N, 3)
        # Sentiment score: pos - neg
        features["FinBERT_sentiment"] = probs[:, 2] - probs[:, 0]
        features["FinBERT_probs"] = probs

    # Baseline: last CLS only
    if "full" in data:
        features["LastCLS"] = data["full"]["last_cls"]

    return features


def reduce_dimension(X, y, method="pca", n_components=10, random_state=42):
    """Dimensionality reduction with train-test split awareness."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    valid = ~np.isnan(y)
    X_valid, y_valid = X_scaled[valid], y[valid]

    if len(X_valid) < 50:
        return np.full(len(X), np.nan), None

    if method == "pca":
        model = PCA(n_components=min(n_components, X_valid.shape[1]),
                    random_state=random_state)
        model.fit(X_valid)
        result = model.transform(X_scaled)
        # Return first component as factor
        return result[:, 0], model

    elif method == "pls":
        n_comp = min(n_components, X_valid.shape[1])
        model = PLSRegression(n_components=n_comp, scale=False)
        model.fit(X_valid, y_valid.reshape(-1, 1))
        result = model.transform(X_scaled)
        if isinstance(result, tuple):
            result = result[0]
        return result[:, 0], model

    elif method == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
            alphas=np.logspace(-4, 0, 20),
            cv=5, max_iter=5000, random_state=random_state,
        )
        model.fit(X_valid, y_valid)
        return model.predict(X_scaled), model

    return np.full(len(X), np.nan), None


def compute_ic_summary(factor, returns, dates=None, method="spearman"):
    """Compute IC statistics."""
    df = pd.DataFrame({
        "factor": factor, "ret": returns,
        "date": pd.to_datetime(dates) if dates is not None else pd.NaT,
    }).dropna()

    if len(df) < 10:
        return {}

    if dates is not None:
        periods = df.groupby(df["date"].dt.to_period("M"))
        ic_vals = []
        for p, g in periods:
            if len(g) >= 10:
                ic, _ = stats.spearmanr(g["factor"], g["ret"])
                ic_vals.append(ic)
        ic_series = pd.Series(ic_vals)
    else:
        ic, _ = stats.spearmanr(df["factor"], df["ret"])
        ic_series = pd.Series([ic])

    return {
        "IC_mean": ic_series.mean(),
        "IC_std": ic_series.std(),
        "ICIR": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        "IC_hit_ratio": (ic_series > 0).mean(),
        "IC_t_stat": ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)) if ic_series.std() > 0 else 0,
        "n_periods": len(ic_series),
        "IC_series": ic_series.values,
    }


def layered_returns(factor, returns, n_groups=5):
    """Compute layered portfolio returns."""
    df = pd.DataFrame({"factor": factor, "ret": returns}).dropna()
    if len(df) < n_groups * 5:
        return None

    df["group"] = pd.qcut(df["factor"], n_groups, labels=False, duplicates="drop") + 1
    result = {}
    for g in range(1, n_groups + 1):
        g_data = df[df["group"] == g]
        result[f"G{g}"] = g_data["ret"].mean() if len(g_data) > 0 else np.nan

    result["LS"] = result[f"G{n_groups}"] - result["G1"]
    result["monotonic"] = all(
        result[f"G{i}"] <= result[f"G{i+1}"]
        for i in range(1, n_groups)
        if not np.isnan(result[f"G{i}"]) and not np.isnan(result[f"G{i+1}"])
    )
    return result


def plot_ic_comparison(ic_results, save_path):
    """Bar chart comparing IC across factors."""
    factors = list(ic_results.keys())
    ic_means = [ic_results[f].get("IC_mean", 0) for f in factors]
    icirs = [ic_results[f].get("ICIR", 0) for f in factors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(factors)))
    ax1.bar(factors, ic_means, color=colors, edgecolor="white")
    ax1.set_title("Rank IC Mean")
    ax1.set_ylabel("IC")
    ax1.tick_params(axis="x", rotation=45)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax2.bar(factors, icirs, color=colors, edgecolor="white")
    ax2.set_title("ICIR (Information Coefficient IR)")
    ax2.set_ylabel("ICIR")
    ax2.tick_params(axis="x", rotation=45)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"[plot] IC comparison saved to {save_path}")


def plot_layered_bar(layered_results, save_path):
    """Grouped bar chart of layered returns."""
    factors = list(layered_results.keys())
    n_groups = 5
    x = np.arange(n_groups)
    width = 0.8 / len(factors)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(factors)))

    for i, (factor, lr) in enumerate(layered_results.items()):
        if lr is None:
            continue
        values = [lr.get(f"G{g}", np.nan) for g in range(1, n_groups + 1)]
        ax.bar(x + i * width, values, width, label=factor, color=colors[i],
               edgecolor="white")

    ax.set_xticks(x + width * (len(factors) - 1) / 2)
    ax.set_xticklabels([f"G{i}" for i in range(1, n_groups + 1)])
    ax.set_ylabel("Average Return")
    ax.set_title("Layered Portfolio Returns (5 Groups)")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"[plot] Layered returns saved to {save_path}")


def plot_layer_importance(data, save_path):
    """Plot IC by BERT layer to analyze layer importance."""
    if "full" not in data:
        return None

    all_cls = data["full"]["all_cls"]  # (N, L, H)
    n_layers = all_cls.shape[1]

    # Use random labels for demonstration (real analysis needs real returns)
    np.random.seed(42)
    y = np.random.randn(all_cls.shape[0])

    layer_ics = []
    for l in range(n_layers):
        layer_factor = all_cls[:, l, 0]  # CLS at position 0
        valid = ~np.isnan(layer_factor) & ~np.isnan(y)
        if valid.sum() > 10:
            ic, _ = stats.spearmanr(layer_factor[valid], y[valid])
            layer_ics.append(ic)
        else:
            layer_ics.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_layers))
    ax.bar(range(1, n_layers + 1), layer_ics, color=colors, edgecolor="white")
    ax.set_xlabel("BERT Layer")
    ax.set_ylabel("|Rank IC|")
    ax.set_title("Layer-wise Predictive Importance")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"[plot] Layer importance saved to {save_path}")

    layer_df = pd.DataFrame({"layer": range(1, n_layers + 1), "ic": layer_ics})
    return layer_df


def plot_sentiment_distribution(data, save_path):
    """Plot FinBERT sentiment score distribution."""
    if "sentiment" not in data or "probabilities" not in data["sentiment"]:
        return

    probs = data["sentiment"]["probabilities"]
    labels = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart of dominant sentiment
    dominant = np.argmax(probs, axis=1)
    counts = [np.sum(dominant == i) for i in range(3)]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90)
    axes[0].set_title("FinBERT Sentiment Classification")

    # Histogram of positive probability
    axes[1].hist(probs[:, 2], bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Positive Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Positive Sentiment Score Distribution")

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"[plot] Sentiment distribution saved to {save_path}")


def plot_title_summary_gap(data, save_path):
    """Plot title-summary semantic gap analysis."""
    if "gap" not in data or "cos_sim" not in data["gap"]:
        return

    cos_sim = data["gap"]["cos_sim"]  # (N, L)
    n_layers = cos_sim.shape[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Average cosine similarity per layer
    mean_cos = cos_sim.mean(axis=0)
    ax1.plot(range(1, n_layers + 1), mean_cos, "o-", color="#3498db", markersize=8)
    ax1.set_xlabel("BERT Layer")
    ax1.set_ylabel("Average Cosine Similarity")
    ax1.set_title("Title-Summary Semantic Similarity by Layer")
    ax1.axhline(y=np.mean(mean_cos), color="red", linestyle="--", alpha=0.5,
                label=f"Mean: {np.mean(mean_cos):.3f}")
    ax1.legend()

    # Distribution
    ax2.hist(cos_sim[:, -1], bins=50, color="#9b59b6", alpha=0.7, edgecolor="white")
    ax2.axvline(x=np.mean(cos_sim[:, -1]), color="red", linestyle="--",
                label=f"Mean: {np.mean(cos_sim[:, -1]):.3f}")
    ax2.set_xlabel("Cosine Similarity (Last Layer)")
    ax2.set_ylabel("Count")
    ax2.set_title("Title-Summary Similarity Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"[plot] Title-summary gap saved to {save_path}")


def generate_latex_table(ic_results, save_path):
    """Generate LaTeX table for IC results."""
    rows = []
    for factor, ic in ic_results.items():
        rows.append(
            f"    {factor} & {ic.get('IC_mean', 0):.4f} & "
            f"{ic.get('ICIR', 0):.3f} & {ic.get('IC_hit_ratio', 0):.2f} & "
            f"{ic.get('n_periods', 0)} \\\\"
        )

    table = f"""\\begin{{table}}[H]
\\centering
\\caption{{Factor IC Comparison}}
\\label{{tab:ic-comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Factor}} & \\textbf{{IC Mean}} & \\textbf{{ICIR}} & \\textbf{{Hit Ratio}} & \\textbf{{N Periods}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(save_path, "w") as f:
        f.write(table)
    print(f"[table] IC table saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("SRTP Comprehensive Analysis")
    print("=" * 60)

    # Load data
    data = load_all_data()
    print(f"Loaded: {list(data.keys())}")

    if not data:
        print("No data available. Run embedding extraction first.")
        return

    # Build features
    features = build_factor_features(data)
    print(f"Features: {list(features.keys())}")

    # Dummy labels for pipeline testing (replace with real returns)
    n_samples = len(next(iter(features.values())))
    np.random.seed(42)
    dummy_returns = 0.02 * np.random.randn(n_samples)
    dummy_dates = pd.date_range("2020-03-02", periods=n_samples, freq="D")

    # Run IC comparison
    print("\n=== IC Analysis ===")
    ic_results = {}
    for name, feat in features.items():
        if feat is None or feat.ndim > 2:
            continue
        # Reduce to single factor
        if feat.ndim == 2 and feat.shape[1] > 1:
            factor, _ = reduce_dimension(feat, dummy_returns, method="pca", n_components=1)
        elif feat.ndim == 2:
            factor = feat[:, 0]
        else:
            factor = feat

        ic = compute_ic_summary(factor, dummy_returns, dummy_dates)
        ic_results[name] = ic
        print(f"  {name:25s}: RankIC={ic.get('IC_mean', 0):.4f}, "
              f"ICIR={ic.get('ICIR', 0):.3f}")

    # Run layered returns
    print("\n=== Layered Returns ===")
    layered_results = {}
    for name, feat in features.items():
        if feat is None or feat.ndim > 2:
            continue
        factor = feat[:, 0] if feat.ndim == 2 else feat
        lr = layered_returns(factor, dummy_returns)
        layered_results[name] = lr
        if lr:
            print(f"  {name:25s}: LS={lr['LS']:.4f}, Monotonic={lr['monotonic']}")

    # Generate plots
    print("\n=== Generating Figures ===")
    plot_ic_comparison(ic_results, FIGURES_DIR / "ic_comparison.png")
    plot_layered_bar(layered_results, FIGURES_DIR / "layered_returns.png")
    if "full" in data:
        plot_layer_importance(data, FIGURES_DIR / "layer_importance.png")
    if "sentiment" in data:
        plot_sentiment_distribution(data, FIGURES_DIR / "sentiment_distribution.png")
    if "gap" in data:
        plot_title_summary_gap(data, FIGURES_DIR / "title_summary_gap.png")

    # Generate LaTeX tables
    generate_latex_table(ic_results, RESULTS_DIR / "ic_table.tex")

    # Save results summary as JSON
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump({
            "ic_results": {k: {kk: vv for kk, vv in v.items()
                               if kk != "IC_series"}
                           for k, v in ic_results.items()},
            "layered_results": layered_results,
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"Figures: {list(FIGURES_DIR.glob('*.png'))}")
    print(f"Results: {list(RESULTS_DIR.glob('*'))}")


if __name__ == "__main__":
    main()
