"""
Final comprehensive PLS pipeline: all FHF variants with rolling training
"""
import json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

EMBED_DIR = Path("/root/autodl-tmp/srtp/embeddings")
DATA_DIR = Path("/root/srtp/data")
RESULTS_DIR = Path("/root/srtp/results")
RESULTS_DIR.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def winsorize(s, lo=0.01, hi=0.99):
    s = np.asarray(s, float)
    l, h = np.nanquantile(s, [lo, hi])
    return np.clip(s, l, h)

def evaluate(name, factor, ret, dates):
    df = pd.DataFrame({'f': factor, 'r': ret, 'd': pd.to_datetime(dates)}).dropna()
    if len(df) < 50: return None
    m_ics = [stats.spearmanr(g['f'], g['r'])[0] for _, g in df.groupby(df['d'].dt.to_period('M')) if len(g) >= 10]
    if not m_ics: return None
    im, sd = np.mean(m_ics), np.std(m_ics)
    t = im / sd * np.sqrt(len(m_ics)) if sd > 0 else 0
    ir = im / sd if sd > 0 else 0
    df['g'] = pd.qcut(df['f'], 5, labels=False, duplicates='drop') + 1
    gr = {g: df[df['g']==g]['r'].mean() for g in range(1,6)}
    ls = gr.get(5, np.nan) - gr.get(1, np.nan)
    mono = all(gr.get(i,-np.inf) <= gr.get(i+1,np.inf) for i in range(1,5))
    return {'name':name,'RankIC':float(im),'ICIR':float(ir),'IC_t':float(t),
            'LS':float(ls),'mono':mono,'n_periods':len(m_ics),'n':len(df),
            'G1':float(gr.get(1,0)),'G5':float(gr.get(5,0))}

def rolling_pls(X, y, dates, n_components=5, window=24):
    dates = pd.to_datetime(dates)
    months = sorted(dates.dt.to_period('M').unique())
    factors = np.full(len(X), np.nan)
    for i, month in enumerate(months):
        train_start = month - pd.offsets.MonthEnd(window)
        train_mask = (dates.dt.to_period('M') >= train_start) & (dates.dt.to_period('M') < month)
        test_mask = dates.dt.to_period('M') == month
        if train_mask.sum() < 100 or test_mask.sum() < 10: continue
        X_tr = X[train_mask.values]; y_tr = y[train_mask.values]
        X_te = X[test_mask.values]
        valid = ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        if len(X_tr) < 100: continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        try:
            nc = min(n_components, X_tr_s.shape[1])
            m = PLSRegression(n_components=nc, scale=False)
            m.fit(X_tr_s, y_tr.reshape(-1,1))
            res = m.transform(X_te_s)
            factors[test_mask.values] = (res[0] if isinstance(res, tuple) else res)[:, 0]
        except: continue
    n_pred = (~np.isnan(factors)).sum()
    return factors, n_pred

log("="*60); log("Final Comprehensive PLS Pipeline"); log("="*60)

# Load
log("Loading embeddings...")
full = np.load(EMBED_DIR / "embeddings_full.npz")
title_e = np.load(EMBED_DIR / "embeddings_title.npz")
summary_e = np.load(EMBED_DIR / "embeddings_summary.npz")
gap = np.load(EMBED_DIR / "embeddings_gap.npz")
sent = np.load(EMBED_DIR / "sentiment_finbert.npz")['probabilities']
log(f"All loaded. Shapes: full={full['all_cls'].shape}")

labels = pd.read_csv(DATA_DIR / "reports_with_labels.csv")
target = winsorize(labels['fwd_excess_5d'].fillna(0).values)
dates = labels['tradable_date']
n = min(len(target), full['all_cls'].shape[0])
log(f"Target: n={n:,} mean={target[:n].mean():.4f} std={target[:n].std():.4f}")

# Build feature sets
features = {
    'FHF-CLS': full['all_cls'][:n, -4:, :].reshape(n, -1),          # Last 4 layers CLS
    'FHF-LayerMix': full['all_cls'][:n].reshape(n, -1),              # All 12 layers CLS
    'FHF-TokenPool': full['all_mean'][:n].reshape(n, -1),            # Mean pooling
    'FHF-Gap': np.column_stack([gap['gap_cls'][:n].mean(axis=1),
                                 gap['cos_sim'][:n]]),               # Title-summary gap
    'LastCLS': full['last_cls'][:n],                                  # Single layer CLS
}
log(f"Features: {[(k, v.shape) for k, v in features.items()]}")

# Run PLS for all
log("\n" + "="*60)
log("PLS ROLLING FACTOR EVALUATION (24-month)")
log("="*60)

results = []
# Baselines
for name, factor in [
    ('FinBERT-Sentiment', sent[:n, 2] - sent[:n, 0]),
    ('FinBERT-PosProb', sent[:n, 2]),
    ('Random', np.random.RandomState(42).randn(n)),
]:
    r = evaluate(name, factor, target[:n], dates[:n])
    if r: results.append(r)

# FHF-PLS factors
for feat_name, X in features.items():
    log(f"\nPLS: {feat_name} ({X.shape[1]} features)...")
    factor, n_pred = rolling_pls(X[:n], target[:n], dates[:n], n_components=5)
    log(f"  Predicted: {n_pred}/{n} ({100*n_pred/n:.0f}%)")
    r = evaluate(f"{feat_name}-PLS", factor, target[:n], dates[:n])
    if r: results.append(r)

# Also PCA for comparison
for feat_name in ['FHF-CLS', 'FHF-LayerMix']:
    X = features[feat_name][:n]
    log(f"\nPCA: {feat_name}...")
    from sklearn.decomposition import PCA as PCA_m
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    factor = PCA_m(1, random_state=42).fit_transform(X_s).flatten()
    r = evaluate(f"{feat_name}-PCA", factor, target[:n], dates[:n])
    if r: results.append(r)

# Print summary
results.sort(key=lambda r: abs(r['RankIC']), reverse=True)
log("\n" + "="*60)
log("FINAL RESULTS SUMMARY")
log("="*60)
log(f"{'Factor':30s} {'RankIC':>8s} {'ICIR':>7s} {'t-stat':>7s} {'LS':>8s} {'Mono':>6s} {'Sig':>4s}")
log("-"*72)
for r in results:
    sig = '***' if abs(r['IC_t'])>2.58 else ('**' if abs(r['IC_t'])>1.96 else ('*' if abs(r['IC_t'])>1.64 else ''))
    log(f"{r['name']:30s} {r['RankIC']:+8.4f} {r['ICIR']:+7.3f} {r['IC_t']:+7.2f} {r['LS']:+8.4f} {str(r['mono']):>6s} {sig:>4s}")

# Best
log(f"\n🏆 Best: {results[0]['name']} (IC={results[0]['RankIC']:.4f}, t={results[0]['IC_t']:.2f})")

# PLS vs PCA comparison
pls = [r for r in results if 'PLS' in r['name']]
pca = [r for r in results if 'PCA' in r['name'] and 'FHF' in r['name']]
fin = [r for r in results if r['name']=='FinBERT-Sentiment']
if pls and pca and fin:
    log(f"\n Method Comparison:")
    log(f"  FinBERT-Sentiment:  IC={fin[0]['RankIC']:.4f}, t={fin[0]['IC_t']:.2f}")
    for r_pls in pls[:3]:
        pca_match = [r for r in pca if r['name'].replace('-PCA','') == r_pls['name'].replace('-PLS','')]
        pca_ic = pca_match[0]['RankIC'] if pca_match else 0
        improvement = (abs(r_pls['RankIC']) - abs(pca_ic)) / max(abs(pca_ic), 0.0001) * 100
        log(f"  {r_pls['name']:30s}: IC={r_pls['RankIC']:.4f} (vs PCA {pca_ic:.4f}, {improvement:+.0f}%)")

# Save
with open(RESULTS_DIR / "final_results.json", 'w') as f:
    json.dump(results, f, indent=2)
log(f"\nResults: {RESULTS_DIR}/final_results.json")
log("Pipeline complete!")
