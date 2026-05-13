"""
======================================================================
SRTP: Enhanced Experiments — Improved Factor Construction
======================================================================
Purpose: Go beyond the baseline FHF factors and find higher-IC signals.

Key improvements tested:
  1. Per-layer PLS analysis (which layers are predictive?)
  2. Optimized PLS n_components via internal CV
  3. Layer-weighted CLS ensembles (learn layer importance)
  4. Gap per-layer analysis
  5. Ridge ensemble (better than IC-weighted)
  6. Interaction features (sentiment × layer, gap × sentiment)
  7. Rolling window optimization (12/18/24/30/36 months)

Run on server:  python src/experiments_enhanced.py
======================================================================
"""
import json
import time
import warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

# ============================================================
# Paths — auto-detect server vs local. Embeddings on autodl-tmp.
# ============================================================
EMBED = None
DATA  = None
RES   = None

for base in [Path("/root/srtp"), Path("/root/autodl-tmp/srtp"),
             Path(__file__).resolve().parent.parent]:
    if (base / "data" / "reports_with_labels.csv").exists():
        PROJ = base
        DATA = PROJ / "data"
        # Embeddings may be on mounted data disk
        for emb_base in [Path("/root/autodl-tmp/srtp/embeddings"),
                         DATA / "embeddings",
                         PROJ / "embeddings"]:
            if (emb_base / "embeddings_full.npz").exists():
                EMBED = emb_base
                break
        RES = PROJ / "results"
        break

if DATA is None or EMBED is None:
    raise FileNotFoundError(
        "Cannot find project root or embeddings. "
        "Check /root/srtp/data/ and /root/autodl-tmp/srtp/embeddings/"
    )

RES.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ws(x, lo=0.01, hi=0.99):
    x = np.asarray(x, float)
    v = ~np.isnan(x)
    if v.sum() == 0: return x
    return np.clip(x, *np.nanquantile(x[v], [lo, hi]))

def stdz(x):
    v = ~np.isnan(x)
    r = np.full_like(x, np.nan, dtype=float)
    if v.sum() == 0: return r
    r[v] = (x[v] - x[v].mean()) / x[v].std()
    return r

# ============================================================
# Evaluation
# ============================================================
def eval_factor(name, factor, ret, dates):
    """Monthly RankIC with full statistics."""
    df = pd.DataFrame({
        'f': np.asarray(factor, float),
        'r': np.asarray(ret, float),
        'd': pd.to_datetime(dates)
    }).dropna()
    if len(df) < 50:
        return None
    m_ics = []
    for _, g in df.groupby(df['d'].dt.to_period('M')):
        if len(g) >= 10:
            ic, _ = stats.spearmanr(g['f'], g['r'])
            m_ics.append(ic)
    if len(m_ics) < 6:
        return None
    im, sd = np.mean(m_ics), np.std(m_ics)
    T = len(m_ics)
    t_val = im / sd * np.sqrt(T) if sd > 0 else 0
    # 5-group layered returns
    try:
        df['g'] = pd.qcut(df['f'], 5, labels=False, duplicates='drop') + 1
    except ValueError:
        df['g'] = 1
    gr = {g: df[df['g'] == g]['r'].mean() for g in range(1, 6)}
    ls = gr.get(5, np.nan) - gr.get(1, np.nan)
    hit = np.mean([ic > 0 for ic in m_ics])
    return {
        'name': name,
        'RankIC': float(im),
        'ICIR': float(im / sd) if sd > 0 else 0,
        'IC_t': float(t_val),
        'LS': float(ls),
        'G1': float(gr.get(1, 0)),
        'G5': float(gr.get(5, 0)),
        'hit_ratio': float(hit),
        'n_periods': T,
        'n': len(df),
    }

# ============================================================
# Rolling PLS with internal CV for n_components
# ============================================================
def rolling_pls_cv(X, y, dates, nc_range=(1, 20), train_window=24,
                   n_folds=3, random_state=42):
    """
    Rolling PLS with internal cross-validation to select optimal n_components
    per training window.
    """
    dates = pd.to_datetime(dates)
    dates_period = dates.to_period('M')
    months = sorted(dates_period.unique())
    factors = np.full(len(X), np.nan)
    nc_history = []  # track chosen n_components

    for month in months:
        ts = month - pd.offsets.MonthEnd(train_window)
        tm = (dates_period >= ts) & (dates_period < month)
        xm = dates_period == month
        if tm.sum() < 100 or xm.sum() < 10:
            continue

        Xt_raw = X[tm]
        yt_raw = y[tm]
        valid = ~np.isnan(yt_raw)
        Xt, yt = Xt_raw[valid], yt_raw[valid]
        if len(Xt) < 100:
            continue

        max_comp = min(nc_range[1], Xt.shape[1], len(Xt) // 10)
        if max_comp < 1:
            continue

        # Internal CV to select n_components
        best_nc = 5  # default
        best_corr = -np.inf
        kf = KFold(n_splits=min(n_folds, len(Xt) // 30), shuffle=True,
                   random_state=random_state)

        for nc in range(max(1, nc_range[0]), max_comp + 1):
            fold_corrs = []
            for tr_idx, va_idx in kf.split(Xt):
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xt[tr_idx])
                Xva = sc.transform(Xt[va_idx])
                try:
                    pls = PLSRegression(n_components=nc, scale=False)
                    pls.fit(Xtr, yt[tr_idx].reshape(-1, 1))
                    pred = pls.predict(Xva).flatten()
                    corr, _ = stats.spearmanr(pred, yt[va_idx])
                    if not np.isnan(corr):
                        fold_corrs.append(corr)
                except Exception:
                    continue
            if fold_corrs:
                avg_corr = np.mean(fold_corrs)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_nc = nc

        # Fit final model with best n_components
        try:
            sc = StandardScaler()
            Xt_s = sc.fit_transform(Xt)
            Xe_s = sc.transform(X[xm])

            pls = PLSRegression(n_components=best_nc, scale=False)
            pls.fit(Xt_s, yt.reshape(-1, 1))
            result = pls.transform(Xe_s)
            if isinstance(result, tuple):
                result = result[0]
            factors[xm] = result[:, 0]
            nc_history.append(best_nc)
        except Exception:
            continue

    n_pred = (~np.isnan(factors)).sum()
    avg_nc = np.mean(nc_history) if nc_history else 5
    log(f"  [PLS-CV] pred={n_pred}/{len(X)} avg_nc={avg_nc:.1f}")
    return factors

# ============================================================
# Experiment 1: Per-Layer Analysis
# ============================================================
def experiment_per_layer(full_emb, labels, dates, target):
    """
    Construct PLS factors for each of the 12 layers individually.
    Test: which layers carry the most predictive signal?
    """
    log("\n" + "=" * 60)
    log("EXP 1: Per-Layer PLS Analysis")
    log("=" * 60)

    all_cls = full_emb['all_cls']  # (N, 12, 768)
    n_layers = all_cls.shape[1]
    results = []

    for layer in range(n_layers):
        # Single layer CLS
        X = all_cls[:, layer, :]  # (N, 768)
        log(f"  Layer {layer + 1}/{n_layers}: X={X.shape}")

        # PCA (unsupervised) — n_components=1
        pca_factor = rolling_pls_cv(
            X, target, dates, nc_range=(1, 1), train_window=24
        )
        r = eval_factor(f"Layer{layer+1:02d}-PCA", pca_factor, target, dates)
        if r:
            results.append(r)
            log(f"    PCA:  IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

        # PLS (supervised)
        pls_factor = rolling_pls_cv(
            X, target, dates, nc_range=(1, 5), train_window=24
        )
        r = eval_factor(f"Layer{layer+1:02d}-PLS", pls_factor, target, dates)
        if r:
            results.append(r)
            log(f"    PLS:  IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Also test: top-N layers combined
    # Find best layers from PLS results, combine
    pls_results = [r for r in results if 'PLS' in r['name']]
    if pls_results:
        pls_sorted = sorted(pls_results, key=lambda x: x['RankIC'], reverse=True)
        top_layers = [int(r['name'].split('-')[0].replace('Layer', '')) - 1
                      for r in pls_sorted[:6]]
        log(f"\n  Top-6 layers by PLS IC: {[l+1 for l in top_layers]}")

        # Combine top-6 layers CLS -> PLS
        X_top6 = np.concatenate([all_cls[:, l, :] for l in top_layers], axis=1)
        log(f"  Top-6 combined: X={X_top6.shape}")
        top6_factor = rolling_pls_cv(
            X_top6, target, dates, nc_range=(1, 10), train_window=24
        )
        r = eval_factor("LayerTop6-PLS", top6_factor, target, dates)
        if r:
            results.append(r)
            log(f"    Top6-PLS: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

        # Combine worst-6 layers (should be less predictive — negative control)
        worst_layers = [int(r['name'].split('-')[0].replace('Layer', '')) - 1
                        for r in pls_sorted[-6:]]
        X_bot6 = np.concatenate([all_cls[:, l, :] for l in worst_layers], axis=1)
        bot6_factor = rolling_pls_cv(
            X_bot6, target, dates, nc_range=(1, 10), train_window=24
        )
        r = eval_factor("LayerBot6-PLS", bot6_factor, target, dates)
        if r:
            results.append(r)
            log(f"    Bot6-PLS: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    return results


# ============================================================
# Experiment 2: Optimized Ensemble Construction
# ============================================================
def experiment_ridge_ensemble(factor_dict, labels, dates, target):
    """
    Use Ridge regression (with rolling training) to learn optimal
    ensemble weights, rather than simple IC-weighted average.

    Also compare: equal-weight, IC-weighted, Ridge, and max-ICIR portfolio.
    """
    log("\n" + "=" * 60)
    log("EXP 2: Optimized Ensemble Construction")
    log("=" * 60)

    # Build factor matrix
    valid_names = []
    factor_matrix = []
    for name, factor in factor_dict.items():
        f = np.asarray(factor, float).copy()
        if not np.all(np.isnan(f)):
            f = stdz(f)
            valid_names.append(name)
            factor_matrix.append(f)

    if len(valid_names) < 2:
        log("  Not enough factors for ensemble")
        return []

    F = np.column_stack(factor_matrix)  # (N, K)
    log(f"  Factor matrix: {F.shape} with {len(valid_names)} factors")

    # --- Method 1: Equal weight ---
    eq_w = np.ones(len(valid_names)) / len(valid_names)
    eq_factor = F @ eq_w
    eq_factor = stdz(eq_factor)

    # --- Method 2: IC-weighted (rolling) ---
    dates_pd = pd.DatetimeIndex(pd.to_datetime(dates))
    months = sorted(dates_pd.to_period('M').unique())
    icw_factor = np.full(F.shape[0], np.nan)

    for i, month in enumerate(months):
        ts = month - pd.offsets.MonthEnd(24)
        tm = (dates_pd.to_period('M') >= ts) & (dates_pd.to_period('M') < month)
        xm = dates_pd.to_period('M') == month
        if tm.sum() < 50 or xm.sum() < 10:
            continue
        F_train = F[tm]
        y_train = target[tm]
        valid = ~np.isnan(y_train) & ~np.isnan(F_train).any(axis=1)
        F_train, y_train = F_train[valid], y_train[valid]
        if len(F_train) < 50:
            continue
        # Compute IC for each factor in training window
        ics = []
        for k in range(F.shape[1]):
            ic, _ = stats.spearmanr(F_train[:, k], y_train)
            ics.append(abs(ic) if not np.isnan(ic) else 0)
        total = sum(ics)
        if total > 0:
            w = np.array(ics) / total
            icw_factor[xm] = F[xm] @ w

    icw_factor = stdz(icw_factor)

    # --- Method 3: Ridge regression ensemble ---
    ridge_factor = np.full(F.shape[0], np.nan)
    ridge_weights_history = []

    for i, month in enumerate(months):
        ts = month - pd.offsets.MonthEnd(24)
        tm = (dates_pd.to_period('M') >= ts) & (dates_pd.to_period('M') < month)
        xm = dates_pd.to_period('M') == month
        if tm.sum() < 100 or xm.sum() < 10:
            continue
        F_train = F[tm]
        y_train = target[tm]
        valid = ~np.isnan(y_train) & ~np.isnan(F_train).any(axis=1)
        F_train, y_train = F_train[valid], y_train[valid]
        if len(F_train) < 100:
            continue
        try:
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 20), fit_intercept=True)
            ridge.fit(F_train, y_train)
            ridge_factor[xm] = ridge.predict(F[xm])
            ridge_weights_history.append(ridge.coef_)
        except Exception:
            continue

    ridge_factor = stdz(ridge_factor)

    results = []
    for name, factor in [
        ("Ensemble-Equal", eq_factor),
        ("Ensemble-ICW", icw_factor),
        ("Ensemble-Ridge", ridge_factor),
    ]:
        r = eval_factor(name, factor, target, dates)
        if r:
            results.append(r)
            log(f"  {name:20s}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Save average Ridge weights for interpretability
    if ridge_weights_history:
        avg_weights = np.mean(ridge_weights_history, axis=0)
        log(f"\n  Avg Ridge weights: {dict(zip(valid_names, avg_weights.round(4)))}")

    return results


# ============================================================
# Experiment 3: Gap Per-Layer + Improved Gap Construction
# ============================================================
def experiment_gap_enhanced(gap_emb, full_emb, labels, dates, target):
    """
    Test gap at each layer (not just last layer).
    Test interaction: gap × sentiment.
    Test different gap formulations.
    """
    log("\n" + "=" * 60)
    log("EXP 3: Enhanced Gap Analysis")
    log("=" * 60)

    results = []

    # Per-layer cosine similarity
    cos_sim = gap_emb['cos_sim']  # (N, 12) — cosine per layer
    gap_cls = gap_emb.get('gap_cls', None)  # (N, 12, 768) if available
    euclidean = gap_emb.get('euclidean_dist', None)  # (N, 12) if available

    # Sentiment for interaction
    sent_path = EMBED / "sentiment_finbert.npz"
    if sent_path.exists():
        sent = np.load(sent_path)['probabilities']
        sent_score = sent[:len(target), 2] - sent[:len(target), 0]  # pos - neg
    else:
        sent_score = np.zeros(len(target))

    n_layers = cos_sim.shape[1]
    log(f"  Gap cos_sim per layer: {cos_sim.shape}")

    # Test: gap at each layer
    best_layer_ic = []
    for layer in range(n_layers):
        # Semantic gap = 1 - cosine
        gap_l = 1.0 - cos_sim[:, layer]
        gap_l = ws(gap_l)
        r = eval_factor(f"Gap-L{layer+1:02d}", gap_l, target, dates)
        if r:
            results.append(r)
            best_layer_ic.append((r['RankIC'], layer))
            if abs(r['RankIC']) > 0.008:
                log(f"    L{layer+1:02d}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Best layer
    if best_layer_ic:
        best_layer_ic.sort(key=lambda x: abs(x[0]), reverse=True)
        best_layer = best_layer_ic[0][1]
        log(f"  Best gap layer: {best_layer+1} (IC={best_layer_ic[0][0]:+.4f})")

    # Test: gap × sentiment interaction
    sent_std = stdz(ws(sent_score))
    for layer in [0, n_layers // 2, n_layers - 1]:  # first, middle, last
        gap_l = 1.0 - cos_sim[:, layer]
        # Interaction: high sentiment + high gap = strongest packaging signal
        gap_x_sent = stdz(ws(gap_l)) * sent_std
        r = eval_factor(f"Gap×Sent-L{layer+1}", gap_x_sent, target, dates)
        if r:
            results.append(r)
            if abs(r['RankIC']) > 0.008:
                log(f"    Gap×Sent L{layer+1}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Test: weighted gap (layer-weighted by importance from per-layer analysis)
    if best_layer_ic:
        top_layers = [l for ic, l in best_layer_ic[:4]]
        gap_top = np.column_stack([1.0 - cos_sim[:, l] for l in top_layers])
        gap_top_pls = rolling_pls_cv(
            gap_top, target, dates, nc_range=(1, 4), train_window=24
        )
        r = eval_factor("Gap-TopLayers-PLS", gap_top_pls, target, dates)
        if r:
            results.append(r)
            log(f"    Gap-TopLayers-PLS: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Test: gap + euclidean distance combined
    if euclidean is not None:
        gap_features = np.column_stack([
            1.0 - cos_sim[:, -1],   # last layer semantic gap
            euclidean[:, -1],        # last layer euclidean distance
        ])
        gap_combined = rolling_pls_cv(
            gap_features, target, dates, nc_range=(1, 3), train_window=24
        )
        r = eval_factor("Gap-Semantic+Euclidean", gap_combined, target, dates)
        if r:
            results.append(r)
            log(f"    Gap-Semantic+Euclidean: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    return results


# ============================================================
# Experiment 4: Rolling Window Optimization
# ============================================================
def experiment_window_optimization(full_emb, labels, dates, target):
    """
    Test different training window lengths for PLS factor construction.
    """
    log("\n" + "=" * 60)
    log("EXP 4: Rolling Window Optimization")
    log("=" * 60)

    # Use last-4-layer CLS as base features
    all_cls = full_emb['all_cls']  # (N, 12, 768)
    X = all_cls[:, -4:, :].reshape(all_cls.shape[0], -1)  # last 4 layers
    log(f"  Features: {X.shape}")

    results = []
    for window in [12, 18, 24, 30, 36]:
        factor = rolling_pls_cv(
            X, target, dates, nc_range=(1, 10), train_window=window
        )
        r = eval_factor(f"Window-{window:02d}M", factor, target, dates)
        if r:
            results.append(r)
            log(f"  Window {window:2d}M: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} "
                f"n_periods={r.get('n_periods', '?')}")

    return results


# ============================================================
# Experiment 5: Layer-Weighted CLS (learn layer importance)
# ============================================================
def experiment_layer_weighted(full_emb, labels, dates, target):
    """
    Instead of equal-weight concatenation of layers, learn layer weights
    via Ridge regression on per-layer CLS factors.

    Approach:
    1. For each layer, construct a single-layer PLS factor
    2. Use Ridge regression to combine these 12 factors
    """
    log("\n" + "=" * 60)
    log("EXP 5: Layer-Weighted Ensemble")
    log("=" * 60)

    all_cls = full_emb['all_cls']  # (N, 12, 768)
    n_layers = all_cls.shape[1]
    dates_pd = pd.DatetimeIndex(pd.to_datetime(dates))
    months = sorted(dates_pd.to_period('M').unique())

    # Step 1: Build per-layer PLS factors for each month
    per_layer_factors = np.full((all_cls.shape[0], n_layers), np.nan)
    for layer in range(n_layers):
        X = all_cls[:, layer, :]  # (N, 768)
        factor = rolling_pls_cv(X, target, dates, nc_range=(1, 5), train_window=24)
        per_layer_factors[:, layer] = factor

    # Step 2: Combine layers via rolling Ridge
    layer_ensemble = np.full(all_cls.shape[0], np.nan)
    weight_history = []

    for month in months:
        ts = month - pd.offsets.MonthEnd(24)
        tm = (dates_pd.to_period('M') >= ts) & (dates_pd.to_period('M') < month)
        xm = dates_pd.to_period('M') == month
        if tm.sum() < 100 or xm.sum() < 10:
            continue

        F_train = per_layer_factors[tm]
        y_train = target[tm]
        valid = ~np.isnan(y_train) & ~np.isnan(F_train).any(axis=1)
        F_train, y_train = F_train[valid], y_train[valid]
        if len(F_train) < 100:
            continue

        try:
            # Standardize each factor column
            sc = StandardScaler()
            F_train_s = sc.fit_transform(F_train)
            F_test_s = sc.transform(per_layer_factors[xm])

            ridge = RidgeCV(alphas=np.logspace(-1, 2, 15), fit_intercept=True)
            ridge.fit(F_train_s, y_train)
            layer_ensemble[xm] = ridge.predict(F_test_s)
            weight_history.append(ridge.coef_)
        except Exception:
            continue

    layer_ensemble = stdz(layer_ensemble)

    # Also test: simple mean of per-layer factors
    mean_factor = stdz(np.nanmean(per_layer_factors, axis=1))

    results = []
    for name, factor in [
        ("LayerWeight-Ridge", layer_ensemble),
        ("LayerWeight-Mean", mean_factor),
    ]:
        r = eval_factor(name, factor, target, dates)
        if r:
            results.append(r)
            log(f"  {name:20s}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    if weight_history:
        avg_w = np.mean(weight_history, axis=0)
        top3 = np.argsort(np.abs(avg_w))[-3:][::-1]
        log(f"\n  Top-3 layers by avg weight: "
            f"{', '.join(f'L{l+1}={avg_w[l]:.3f}' for l in top3)}")

    return results


# ============================================================
# Experiment 6: Multi-Horizon Factor
# ============================================================
def experiment_multihorizon(full_emb, labels, dates):
    """
    Build factors targeting different horizons (1d, 5d, 10d, 20d),
    then combine them. Different layers may be optimal for different horizons.
    """
    log("\n" + "=" * 60)
    log("EXP 6: Multi-Horizon Factors")
    log("=" * 60)

    all_cls = full_emb['all_cls']
    X = all_cls[:, -4:, :].reshape(all_cls.shape[0], -1)
    results = []

    # Build factors for each horizon
    horizon_factors = {}
    for horizon, col in [(1, 'fwd_excess_1d'), (5, 'fwd_excess_5d'),
                          (10, 'fwd_excess_10d'), (20, 'fwd_excess_20d')]:
        if col in labels.columns:
            target_h = ws(labels[col].fillna(0).values[:len(dates)])
            factor = rolling_pls_cv(X, target_h, dates, nc_range=(1, 10), train_window=24)
            horizon_factors[f"H{horizon}d"] = stdz(factor)
            r = eval_factor(f"PLS-H{horizon}d", factor, target_h, dates)
            if r:
                results.append(r)
                log(f"  H{horizon}d target: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Combine all horizons -> predict 5d return
    target5 = ws(labels['fwd_excess_5d'].fillna(0).values[:len(dates)])
    F_horizons = np.column_stack([v for v in horizon_factors.values()])
    combined = rolling_pls_cv(F_horizons, target5, dates, nc_range=(1, 3), train_window=24)
    r = eval_factor("MultiHorizon-PLS", combined, target5, dates)
    if r:
        results.append(r)
        log(f"  MultiHorizon combined: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("SRTP Enhanced Experiments")
    log("=" * 60)

    # Load data
    log("\nLoading data...")
    labels = pd.read_csv(DATA / "reports_with_labels.csv")
    labels['report_date'] = pd.to_datetime(labels['report_date'])
    labels['tradable_date'] = pd.to_datetime(labels['tradable_date'])

    full = np.load(EMBED / "embeddings_full.npz")
    gap  = np.load(EMBED / "embeddings_gap.npz")

    n = min(len(labels), full['all_cls'].shape[0])
    log(f"  Samples: {n:,}")

    labels = labels.iloc[:n]
    target5 = ws(labels['fwd_excess_5d'].fillna(0).values[:n])
    dates = labels['tradable_date'].values[:n]

    # Truncate embeddings to match labels
    full_trunc = {k: full[k][:n] for k in full.files}
    gap_trunc = {k: gap[k][:n] for k in gap.files}

    # Sentiment baseline
    sent_path = EMBED / "sentiment_finbert.npz"
    if sent_path.exists():
        sent = np.load(sent_path)['probabilities']
        sent_score = sent[:n, 2] - sent[:n, 0]  # pos - neg
        r = eval_factor("FinBERT-Sentiment", sent_score, target5, dates)
        if r:
            log(f"\n  Baseline Sentiment: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    all_results = []

    # Run experiments
    for exp_fn, exp_name in [
        (experiment_per_layer, "EXP1_PerLayer"),
        (experiment_gap_enhanced, "EXP3_GapEnhanced"),
        (experiment_window_optimization, "EXP4_WindowOptimization"),
        (experiment_layer_weighted, "EXP5_LayerWeighted"),
        (experiment_multihorizon, "EXP6_MultiHorizon"),
    ]:
        try:
            log(f"\n{'='*40}")
            log(f"Running {exp_name}...")
            if exp_name in ("EXP3_GapEnhanced",):
                results = exp_fn(gap_trunc, full_trunc, labels, dates, target5)
            elif exp_name in ("EXP6_MultiHorizon",):
                results = exp_fn(full_trunc, labels, dates)
            elif exp_name in ("EXP4_WindowOptimization", "EXP5_LayerWeighted", "EXP1_PerLayer"):
                results = exp_fn(full_trunc, labels, dates, target5)
            else:
                results = exp_fn(full_trunc, labels, dates, target5)
            all_results.extend(results or [])
        except Exception as e:
            log(f"  ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # Run EXP2 (Ensemble) last — needs factor dict from other experiments
    try:
        # Build factor dict for ensemble
        all_cls = full_trunc['all_cls']
        X_last4 = all_cls[:, -4:, :].reshape(all_cls.shape[0], -1)
        X_all12 = all_cls.reshape(all_cls.shape[0], -1)

        # Quick PLS factors
        f_cls_pls = rolling_pls_cv(X_last4, target5, dates, nc_range=(1, 10), train_window=24)
        f_all_pls = rolling_pls_cv(X_all12, target5, dates, nc_range=(1, 10), train_window=24)

        # Gap factor
        cos_sim = gap_trunc['cos_sim']
        eucl = gap_trunc.get('euclidean_dist', None)
        gap_feat = np.column_stack([1.0 - cos_sim[:, -1],
                                     eucl[:, -1] if eucl is not None else np.zeros(len(cos_sim))])
        f_gap = rolling_pls_cv(gap_feat, target5, dates, nc_range=(1, 3), train_window=24)

        factor_dict = {
            'FHF-CLS-PLS': stdz(f_cls_pls),
            'FHF-AllCLS-PLS': stdz(f_all_pls),
            'FHF-Gap-PLS': stdz(f_gap),
            'Sentiment': stdz(sent_score),
        }
        results = experiment_ridge_ensemble(factor_dict, labels, dates, target5)
        all_results.extend(results or [])
    except Exception as e:
        log(f"  ERROR in EXP2_Ensemble: {e}")

    # Save results
    log("\n" + "=" * 60)
    log("SAVING RESULTS")
    log("=" * 60)

    # Sort by abs(IC)
    all_results.sort(key=lambda x: abs(x['RankIC']), reverse=True)

    # Print top results
    log("\nTOP RESULTS:")
    log(f"{'Name':<30s} {'RankIC':>8s} {'ICIR':>8s} {'t-val':>8s} {'Hit':>6s}")
    log("-" * 65)
    for r in all_results[:20]:
        log(f"{r['name']:<30s} {r['RankIC']:>+8.4f} {r['ICIR']:>8.3f} "
            f"{r['IC_t']:>8.2f} {r['hit_ratio']:>6.1%}")

    # Save JSON
    out_path = RES / "enhanced_experiments.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nFull results saved to {out_path}")

    # Also save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RES / "enhanced_experiments.csv", index=False)
    log(f"CSV saved to {RES / 'enhanced_experiments.csv'}")

    log("\nExperiments complete!")


if __name__ == "__main__":
    main()
