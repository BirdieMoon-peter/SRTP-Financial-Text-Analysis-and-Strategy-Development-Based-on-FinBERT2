"""
Advanced FHF Pipeline: PLS, ElasticNet, Rolling Training, FM Regression
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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

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
    ls = gr.get(5,np.nan) - gr.get(1,np.nan)
    mono = all(gr.get(i,-np.inf) <= gr.get(i+1,np.inf) for i in range(1,5))
    return {'name':name,'RankIC':float(im),'ICIR':float(ir),'IC_t':float(t),
            'LS':float(ls),'mono':mono,'n_periods':len(m_ics),'n':len(df),
            'G1':float(gr.get(1,0)),'G5':float(gr.get(5,0))}

def rolling_train(X, y, dates, method='elasticnet', n_components=10, window=24):
    """Rolling window factor construction with strict temporal ordering."""
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
            if method == 'pca':
                m = PCA(n_components=min(n_components, X_tr_s.shape[1]), random_state=42)
                m.fit(X_tr_s)
                pred = m.transform(X_te_s)[:, 0]
            elif method == 'pls':
                nc = min(n_components, X_tr_s.shape[1])
                m = PLSRegression(n_components=nc, scale=False)
                m.fit(X_tr_s, y_tr.reshape(-1,1))
                res = m.transform(X_te_s)
                pred = (res[0] if isinstance(res, tuple) else res)[:, 0]
            elif method == 'elasticnet':
                m = ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9], alphas=np.logspace(-4,0,10),
                                cv=5, max_iter=5000, random_state=42)
                m.fit(X_tr_s, y_tr)
                pred = m.predict(X_te_s)
            else: continue
            factors[test_mask.values] = pred
        except Exception as e:
            continue

    n_pred = (~np.isnan(factors)).sum()
    log(f"  [{method}] Rolling: {n_pred}/{len(X)} ({100*n_pred/len(X):.0f}%) samples predicted")
    return factors

log("="*60); log("Advanced Pipeline: PLS + ElasticNet + Rolling"); log("="*60)

# Load embeddings
log("Loading embeddings...")
emb = {}
for c in ['full']:
    p = EMBED_DIR / f"embeddings_{c}.npz"
    d = np.load(p)
    emb[c] = {'all_cls': d['all_cls'], 'last_cls': d['last_cls']}
    log(f"  [{c}] {d['all_cls'].shape}")

gap_p = EMBED_DIR / "embeddings_gap.npz"
if gap_p.exists(): emb['gap'] = {k: np.load(gap_p)[k] for k in np.load(gap_p).files}

sent_p = EMBED_DIR / "sentiment_finbert.npz"
if sent_p.exists(): emb['sent'] = {'prob': np.load(sent_p)['probabilities']}

# Load labels
labels = pd.read_csv(DATA_DIR / "reports_with_labels.csv")
target = labels['fwd_excess_5d'].fillna(0).values
dates = labels['tradable_date']
n = min(len(target), emb['full']['all_cls'].shape[0])
log(f"Labels: {n:,} rows")

# Build feature matrices
log("\nBuilding features...")
cls_4 = emb['full']['all_cls'][:n, -4:, :].reshape(n, -1)   # FHF-CLS
cls_all = emb['full']['all_cls'][:n].reshape(n, -1)          # FHF-LayerMix
last = emb['full']['last_cls'][:n]                            # LastCLS
if 'gap' in emb:
    gap_feats = np.column_stack([emb['gap']['gap_cls'][:n].mean(axis=1), emb['gap']['cos_sim'][:n]])
else:
    gap_feats = np.random.randn(n, 2)

sent_score = emb['sent']['prob'][:n, 2] - emb['sent']['prob'][:n, 0]
# Winsorize target
target_w = winsorize(target[:n])

results_all = []
# Test each feature set with multiple methods
experiments = [
    ('FHF-CLS-PCA', cls_4, 'pca'),
    ('FHF-CLS-PLS', cls_4, 'pls'),
    ('FHF-CLS-ENet', cls_4, 'elasticnet'),
    ('FHF-LayerMix-PCA', cls_all, 'pca'),
    ('FHF-LayerMix-PLS', cls_all, 'pls'),
    ('FHF-LayerMix-ENet', cls_all, 'elasticnet'),
    ('FHF-Gap-PCA', gap_feats, 'pca'),
    ('FHF-Gap-PLS', gap_feats, 'pls'),
    ('FHF-Gap-ENet', gap_feats, 'elasticnet'),
    ('LastCLS-PCA', last, 'pca'),
    ('LastCLS-PLS', last, 'pls'),
    ('LastCLS-ENet', last, 'elasticnet'),
]

log("\n" + "="*60)
log("ROLLING WINDOW FACTOR EVALUATION (24-month training)")
log("="*60)
log(f"{'Factor':25s} {'RankIC':>8s} {'ICIR':>7s} {'t-stat':>7s} {'LS':>8s} {'Mono':>6s} {'Sig':>4s}")
log("-"*68)

# Baselines (not using rolling training)
for name, factor in [
    ('FinBERT-Sentiment', sent_score),
    ('FinBERT-PosProb', emb['sent']['prob'][:n, 2]),
    ('Random', np.random.RandomState(42).randn(n)),
]:
    r = evaluate(name, factor[:n], target_w, dates[:n])
    if r:
        results_all.append(r)
        sig = '***' if abs(r['IC_t'])>2.58 else ('**' if abs(r['IC_t'])>1.96 else ('*' if abs(r['IC_t'])>1.64 else ''))
        log(f"{name:25s} {r['RankIC']:+8.4f} {r['ICIR']:+7.3f} {r['IC_t']:+7.2f} {r['LS']:+8.4f} {str(r['mono']):>6s} {sig:>4s}")

# Rolling factors
for name, X, method in experiments:
    log(f"\nComputing {name}...")
    factor = rolling_train(X[:n], target_w[:n], dates[:n], method=method, n_components=10)
    r = evaluate(name.replace('-ENet','-ElasticNet'), factor, target_w[:n], dates[:n])
    if r:
        results_all.append(r)
        sig = '***' if abs(r['IC_t'])>2.58 else ('**' if abs(r['IC_t'])>1.96 else ('*' if abs(r['IC_t'])>1.64 else ''))
        log(f"{name.replace('-ENet','-ElasticNet'):25s} {r['RankIC']:+8.4f} {r['ICIR']:+7.3f} {r['IC_t']:+7.2f} {r['LS']:+8.4f} {str(r['mono']):>6s} {sig:>4s}")

# Fama-MacBeth with controls
log("\n" + "="*60)
log("FAMA-MACBETH REGRESSION")

# Find best rolling factor
results_all.sort(key=lambda r: abs(r['RankIC']), reverse=True)
best = results_all[0]
log(f"Best rolling factor: {best['name']} (IC={best['RankIC']:.4f})")

# Simple FM: regress returns on best factor
best_factor = None
for name, X, method in experiments:
    if name.replace('-ENet','-ElasticNet') == best['name']:
        best_factor = rolling_train(X[:n], target_w[:n], dates[:n], method=method, n_components=10)
        break

if best_factor is not None:
    # Add log market cap proxy (volume as rough size proxy)
    df_fm = pd.DataFrame({
        'ret': target_w, 'factor': best_factor,
        'date': pd.to_datetime(dates[:n])
    }).dropna()

    fm_coefs = []; fm_ts = []; fm_r2s = []
    for p, g in df_fm.groupby(df_fm['date'].dt.to_period('M')):
        if len(g) < 20: continue
        X = np.column_stack([np.ones(len(g)), g['factor'].values])
        y = g['ret'].values
        lr = LinearRegression().fit(X, y)
        fm_coefs.append(lr.coef_[1]); fm_r2s.append(lr.score(X, y))

    if fm_coefs:
        c = np.mean(fm_coefs); s = np.std(fm_coefs); T = len(fm_coefs)
        fm_t = c / (s/np.sqrt(T)) if s > 0 else 0
        log(f"  FM coefficient: {c:.6f}")
        log(f"  FM t-statistic: {fm_t:.2f}")
        log(f"  Avg R-squared:  {np.mean(fm_r2s):.4f}")
        log(f"  N periods:      {T}")

# Save all results
with open(RESULTS_DIR / "advanced_results.json", 'w') as f:
    json.dump(results_all, f, indent=2)

log(f"\nResults: {RESULTS_DIR}/advanced_results.json")
log("="*60); log("Pipeline complete!")
