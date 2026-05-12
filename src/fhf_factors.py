"""
SRTP: FinBERT Hidden Layer Text Factor Research
FHF Factor Construction Module
==============================================
Implements the FinBERT Hidden Factor (FHF) algorithm family:
FHF-CLS, FHF-LayerMix, FHF-TokenPool, FHF-Gap, FHF-Ensemble.

Dimensionality reduction methods: PCA, PLS, ElasticNet, LightGBM.
All with strict rolling training to prevent look-ahead bias.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARNING] lightgbm not available, LightGBM method disabled")

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_embeddings():
    """Load all cached embeddings and sentiment outputs."""
    print("[load] Loading cached embeddings...")
    data = {}

    for config in ["title", "summary", "full"]:
        try:
            d = np.load(EMBED_DIR / f"embeddings_{config}.npz")
            meta_path = EMBED_DIR / f"embeddings_{config}_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            data[config] = {
                "last_cls": d["last_cls"],       # (N, 768)
                "all_cls": d["all_cls"],         # (N, L, 768)
                "all_mean": d["all_mean"],       # (N, L, 768)
                "n_layers": meta["n_layers"],
                "hidden_dim": meta["hidden_dim"],
            }
            print(f"  [{config}] loaded: last_cls {d['last_cls'].shape}, "
                  f"all_cls {d['all_cls'].shape}")
        except FileNotFoundError:
            print(f"  [{config} WARNING: not found]")

    # Gap features
    try:
        d = np.load(EMBED_DIR / "embeddings_gap.npz")
        data["gap"] = {
            "gap_cls": d["gap_cls"],
            "gap_mean": d["gap_mean"],
            "cos_sim": d["cos_sim"],
            "euclidean_dist": d["euclidean_dist"],
        }
        print(f"  [gap] loaded")
    except FileNotFoundError:
        print("  [gap WARNING: not found]")

    # FinBERT sentiment
    try:
        d = np.load(EMBED_DIR / "sentiment_finbert.npz")
        data["sentiment"] = {
            "logits": d["logits"],
            "probabilities": d["probabilities"],
        }
        print(f"  [sentiment] loaded: probabilities {d['probabilities'].shape}")
    except FileNotFoundError:
        print("  [sentiment WARNING: not found]")

    return data


# ---------------------------------------------------------------------------
# Feature builders: construct input matrices from embeddings
# ---------------------------------------------------------------------------

def build_cls_features(emb_data, layers=None):
    """Extract CLS features from specified layers. Default: last 4 layers."""
    all_cls = emb_data["all_cls"]  # (N, L, H)
    L = all_cls.shape[1]
    if layers is None:
        layers = list(range(max(0, L - 4), L))  # last 4 layers
    feats = all_cls[:, layers, :]  # (N, len(layers), H)
    return feats.reshape(feats.shape[0], -1)  # (N, len(layers)*H)


def build_last_cls(emb_data):
    """Extract only last layer CLS."""
    return emb_data["last_cls"]  # (N, H)


def build_layer_mix_features(emb_data):
    """Concatenate CLS from all layers."""
    all_cls = emb_data["all_cls"]  # (N, L, H)
    return all_cls.reshape(all_cls.shape[0], -1)  # (N, L*H)


def build_mean_pool_features(emb_data, layers=None):
    """Extract mean pooling features from specified layers."""
    all_mean = emb_data["all_mean"]  # (N, L, H)
    L = all_mean.shape[1]
    if layers is None:
        layers = list(range(L))
    feats = all_mean[:, layers, :]
    return feats.reshape(feats.shape[0], -1)


def build_gap_features(gap_data):
    """Build title-summary gap features."""
    feats = []
    if "gap_cls" in gap_data:
        # (N, L, H) -> average over layers
        feats.append(gap_data["gap_cls"].mean(axis=1))  # (N, H)
    if "gap_mean" in gap_data:
        feats.append(gap_data["gap_mean"].mean(axis=1))
    if "cos_sim" in gap_data:
        feats.append(gap_data["cos_sim"])  # (N, L)
    if "euclidean_dist" in gap_data:
        feats.append(gap_data["euclidean_dist"])  # (N, L)
    return np.concatenate(feats, axis=1) if feats else np.array([])


def build_combined_features(embed_data, include_gap=True):
    """Build comprehensive feature set combining all representations."""
    features = []
    feature_names = []

    # Last CLS
    feats = build_last_cls(embed_data["full"])
    features.append(feats)
    feature_names.extend([f"full_last_cls_{i}" for i in range(feats.shape[1])])

    # Multi-layer CLS
    feats = build_cls_features(embed_data["full"])
    features.append(feats)
    feature_names.extend([f"full_cls_{i}" for i in range(feats.shape[1])])

    # Title last CLS
    feats = build_last_cls(embed_data["title"])
    features.append(feats)
    feature_names.extend([f"title_last_cls_{i}" for i in range(feats.shape[1])])

    # Summary last CLS
    feats = build_last_cls(embed_data["summary"])
    features.append(feats)
    feature_names.extend([f"summary_last_cls_{i}" for i in range(feats.shape[1])])

    # Mean pooling
    feats = build_mean_pool_features(embed_data["full"])
    features.append(feats)
    feature_names.extend([f"full_mean_{i}" for i in range(feats.shape[1])])

    if include_gap and "gap" in embed_data:
        feats = build_gap_features(embed_data["gap"])
        if feats.size > 0:
            features.append(feats)
            feature_names.extend([f"gap_{i}" for i in range(feats.shape[1])])

    X = np.concatenate(features, axis=1)
    return X, feature_names


# ---------------------------------------------------------------------------
# Dimensionality reduction methods
# ---------------------------------------------------------------------------

class DimensionReducer:
    """Base class for dimensionality reduction with rolling training."""

    def __init__(self, method="pca", n_components=20, random_state=42, **kwargs):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        X_scaled = self.scaler_.fit_transform(X)

        if self.method == "pca":
            self.model_ = PCA(n_components=min(self.n_components, X.shape[1]),
                              random_state=self.random_state)
            self.model_.fit(X_scaled)

        elif self.method == "pls":
            if y is None:
                raise ValueError("PLS requires labels y")
            n_comp = min(self.n_components, X.shape[1])
            self.model_ = PLSRegression(n_components=n_comp, scale=False)
            self.model_.fit(X_scaled, y.reshape(-1, 1) if y.ndim == 1 else y)

        elif self.method == "elasticnet":
            if y is None:
                raise ValueError("ElasticNet requires labels y")
            self.model_ = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                alphas=np.logspace(-4, 0, 20),
                cv=5, max_iter=5000, random_state=self.random_state,
                **self.kwargs
            )
            self.model_.fit(X_scaled, y)

        elif self.method == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("lightgbm not available")
            if y is None:
                raise ValueError("LightGBM requires labels y")
            self.model_ = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, num_leaves=15,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=self.random_state, verbose=-1,
                **self.kwargs
            )
            self.model_.fit(X_scaled, y)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, X):
        X_scaled = self.scaler_.transform(X)

        if self.method == "pca":
            return self.model_.transform(X_scaled)

        elif self.method == "pls":
            result = self.model_.transform(X_scaled)
            # PLS returns (X, Y) tuple
            if isinstance(result, tuple):
                return result[0]
            return result

        elif self.method == "elasticnet":
            return self.model_.predict(X_scaled).reshape(-1, 1)

        elif self.method == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("lightgbm not available")
            return self.model_.predict(X_scaled).reshape(-1, 1)

        return X_scaled

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        # PLS fit_transform returns tuple, extract X scores
        if isinstance(result, tuple):
            result = result[0]
        return result

    def get_feature_importance(self):
        """Return feature importance if available."""
        if self.method == "elasticnet":
            return np.abs(self.model_.coef_)
        elif self.method == "lightgbm":
            return self.model_.feature_importances_
        elif self.method == "pca":
            return self.model_.explained_variance_ratio_
        return None


# ---------------------------------------------------------------------------
# Rolling factor construction
# ---------------------------------------------------------------------------

def rolling_factor_construction(X, y, dates, method="pca",
                                 train_window=24, test_window=1,
                                 frequency="M", **kwargs):
    """
    Rolling window factor construction with strict temporal ordering.

    Parameters:
    - X: feature matrix (N, D)
    - y: target labels (N,) - e.g., fwd_ret_5d
    - dates: datetime Series (N,)
    - train_window: training window in months
    - test_window: prediction period in months
    - frequency: "M" for monthly re-estimation
    """
    dates = pd.to_datetime(dates)
    unique_months = sorted(dates.dt.to_period("M").unique())

    factors = np.full(len(X), np.nan)
    reducer_models = []

    for i, month in enumerate(unique_months):
        # Training period: previous train_window months
        train_start = month - pd.offsets.MonthEnd(train_window)
        train_mask = (dates.dt.to_period("M") >= train_start) & \
                     (dates.dt.to_period("M") < month)
        test_mask = dates.dt.to_period("M") == month

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train = X[train_mask.values]
        y_train = y[train_mask.values]

        # Drop NaN labels
        valid = ~np.isnan(y_train)
        X_train, y_train = X_train[valid], y_train[valid]

        if len(X_train) < 50:
            continue

        # Fit reducer on training data
        reducer = DimensionReducer(method=method, **kwargs)
        try:
            reducer.fit(X_train, y_train)
            X_test = X[test_mask.values]
            factors[test_mask.values] = reducer.transform(X_test).flatten()
            reducer_models.append({"month": str(month), "model": reducer})
        except Exception as e:
            print(f"  [WARNING] Month {month}: {e}")
            continue

    n_predicted = (~np.isnan(factors)).sum()
    print(f"[rolling] {method}: {n_predicted}/{len(X)} samples with factors "
          f"({100*n_predicted/len(X):.1f}%)")

    return factors, reducer_models


# ---------------------------------------------------------------------------
# Factor aggregation and processing
# ---------------------------------------------------------------------------

def aggregate_reports_to_stock(report_scores, report_dates, stock_codes,
                                tradable_dates, halflife_days=5):
    """
    Aggregate multiple report scores to stock level using time-decay weighting.

    Parameters:
    - report_scores: array of per-report factor scores
    - report_dates: per-report dates
    - stock_codes: per-report stock codes
    - tradable_dates: per-report first tradable dates
    - halflife_days: half-life for time decay weighting
    """
    df = pd.DataFrame({
        "stock_code": stock_codes,
        "report_date": pd.to_datetime(report_dates),
        "tradable_date": pd.to_datetime(tradable_dates),
        "score": report_scores,
    })

    # Time decay weight: more recent reports get higher weight
    results = []
    for (stock, tdate), group in df.groupby(["stock_code", "tradable_date"]):
        if len(group) == 1:
            results.append({
                "stock_code": stock,
                "tradable_date": tdate,
                "score_mean": group["score"].iloc[0],
                "score_median": group["score"].iloc[0],
                "n_reports": 1,
            })
        else:
            # Time-decay weighted mean
            ref_date = tdate
            ages = (ref_date - pd.to_datetime(group["report_date"])).dt.days.values
            ages = np.maximum(ages, 0)
            weights = np.exp(-np.log(2) * ages / halflife_days)
            weights = weights / weights.sum()

            results.append({
                "stock_code": stock,
                "tradable_date": tdate,
                "score_mean": np.average(group["score"], weights=weights),
                "score_median": np.median(group["score"]),
                "n_reports": len(group),
            })

    return pd.DataFrame(results)


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at given quantiles."""
    lo, hi = np.nanquantile(series, [lower, upper])
    return np.clip(series, lo, hi)


def standardize(series):
    """Z-score standardization."""
    mu, sigma = np.nanmean(series), np.nanstd(series)
    if sigma < 1e-10:
        return np.zeros_like(series)
    return (series - mu) / sigma


def industry_neutralize(factor, industry_dummies):
    """Regress factor on industry dummies, return residuals."""
    from sklearn.linear_model import LinearRegression
    valid = ~np.isnan(factor)
    if valid.sum() < 10:
        return factor
    X = industry_dummies[valid].values if hasattr(industry_dummies, 'values') else industry_dummies[valid]
    y = factor[valid]
    lr = LinearRegression().fit(X, y)
    residuals = factor.copy()
    residuals[valid] = y - lr.predict(X)
    return residuals


# ---------------------------------------------------------------------------
# FHF Factor Family
# ---------------------------------------------------------------------------

def build_fhf_cls(embed_data, labels, dates, method="pca", n_components=20,
                  train_window=24, **kwargs):
    """Build FHF-CLS factor: last 4 layer CLS concatenated + dimensionality reduction."""
    X = build_cls_features(embed_data["full"])  # last 4 layers
    print(f"[FHF-CLS] Features: {X.shape}")
    factors, models = rolling_factor_construction(
        X, labels, dates, method=method, n_components=n_components,
        train_window=train_window, **kwargs
    )
    return factors, models, X.shape


def build_fhf_layermix(embed_data, labels, dates, method="elasticnet",
                        train_window=24, **kwargs):
    """Build FHF-LayerMix factor: all layers CLS with learned weights."""
    X = build_layer_mix_features(embed_data["full"])  # all 12 layers
    print(f"[FHF-LayerMix] Features: {X.shape}")
    factors, models = rolling_factor_construction(
        X, labels, dates, method=method, n_components=20,
        train_window=train_window, **kwargs
    )
    return factors, models, X.shape


def build_fhf_tokenpool(embed_data, labels, dates, method="lightgbm",
                         train_window=24, **kwargs):
    """Build FHF-TokenPool factor: mean pooling from all layers."""
    X = build_mean_pool_features(embed_data["full"])  # all 12 layers mean pool
    print(f"[FHF-TokenPool] Features: {X.shape}")
    factors, models = rolling_factor_construction(
        X, labels, dates, method=method, n_components=20,
        train_window=train_window, **kwargs
    )
    return factors, models, X.shape


def build_fhf_gap(embed_data, labels, dates, method="elasticnet",
                   train_window=24, **kwargs):
    """Build FHF-Gap factor: title-summary semantic difference."""
    if "gap" not in embed_data:
        print("[FHF-Gap] Gap features not available")
        return None, None, None
    X = build_gap_features(embed_data["gap"])
    if X.size == 0:
        print("[FHF-Gap] No gap features")
        return None, None, None
    print(f"[FHF-Gap] Features: {X.shape}")
    factors, models = rolling_factor_construction(
        X, labels, dates, method=method, n_components=10,
        train_window=train_window, **kwargs
    )
    return factors, models, X.shape


def build_fhf_ensemble(factor_dict, weights=None):
    """Build FHF-Ensemble: weighted combination of multiple factors."""
    valid_factors = {}
    for name, factor in factor_dict.items():
        if factor is not None and not np.all(np.isnan(factor)):
            # Standardize first
            valid_factors[name] = standardize(factor)

    if not valid_factors:
        return None

    # Equal weight if not specified
    if weights is None:
        weights = {k: 1.0 / len(valid_factors) for k in valid_factors}

    result = np.zeros_like(list(valid_factors.values())[0])
    total_w = 0
    for name, factor in valid_factors.items():
        w = weights.get(name, 0)
        result += w * factor
        total_w += w

    if total_w > 0:
        result = result / total_w

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SRTP FHF Factor Construction")
    print("=" * 60)

    # Setup directories
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Load embeddings
    embed_data = load_embeddings()

    print("\nFHF factor construction requires:")
    print("  1. Embeddings (loaded above)")
    print("  2. Labels (from build_labels.py with CSMAR market data)")
    print("  3. Rolling training parameters")
    print()
    print("Factor family:")
    print("  FHF-CLS:      Last 4 layer CLS -> PCA/PLS/ElasticNet")
    print("  FHF-LayerMix: All layers CLS -> ElasticNet weighted")
    print("  FHF-TokenPool: Token mean pooling -> LightGBM")
    print("  FHF-Gap:      Title-summary semantic gap")
    print("  FHF-Ensemble: Weighted combination")

    # Quick test with random labels if no real labels available
    np.random.seed(42)
    n_samples = embed_data.get("full", {}).get("last_cls", np.empty((0,))).shape[0]
    if n_samples > 0:
        print(f"\n[TEST] Running factor construction on {n_samples} samples with random labels...")
        mock_labels = np.random.randn(n_samples)
        mock_dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        mock_dates = pd.Series(mock_dates)

        for method_name, build_fn in [
            ("FHF-CLS", build_fhf_cls),
            ("FHF-LayerMix", build_fhf_layermix),
            ("FHF-TokenPool", build_fhf_tokenpool),
        ]:
            print(f"\n--- {method_name} (pca) ---")
            factors, models, shape = build_fn(
                embed_data, mock_labels, mock_dates, method="pca",
                n_components=10, train_window=24
            )
            if factors is not None:
                print(f"  Factor shape: {factors.shape}")
                print(f"  Factor mean/std: {np.nanmean(factors):.4f}/{np.nanstd(factors):.4f}")
                print(f"  Coverage: {(~np.isnan(factors)).sum()}/{len(factors)}")


if __name__ == "__main__":
    main()
