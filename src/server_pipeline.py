"""
Server-side full pipeline: factor construction + evaluation
"""
import json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

EMBED_DIR = Path("/root/autodl-tmp/srtp/embeddings")
DATA_DIR = Path("/root/srtp/data")
RESULTS_DIR = Path("/root/srtp/results")
RESULTS_DIR.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def evaluate_factor(factor, returns, dates, name):
    df = pd.DataFrame({'f': np.asarray(factor,float), 'r': np.asarray(returns,float),
                       'd': pd.to_datetime(dates)}).dropna()
    if len(df) < 30: return None
    monthly = []
    for p, g in df.groupby(df['d'].dt.to_period('M')):
        if len(g) >= 10:
            ic, _ = stats.spearmanr(g['f'], g['r'])
            monthly.append(ic)
    if not monthly: return None
    ic_m = np.mean(monthly); ic_s = np.std(monthly)
    ic_t = ic_m/ic_s*np.sqrt(len(monthly)) if ic_s > 0 else 0
    df['g'] = pd.qcut(df['f'], 5, labels=False, duplicates='drop') + 1
    gr = {g: df[df['g']==g]['r'].mean() for g in range(1,6)}
    ls = gr.get(5,np.nan) - gr.get(1,np.nan)
    mono = all(gr.get(i,-np.inf) <= gr.get(i+1,np.inf) for i in range(1,5))
    return {'name':name,'RankIC':float(ic_m),'ICIR':float(ic_m/ic_s) if ic_s>0 else 0,
            'IC_t':float(ic_t),'LS':float(ls),'mono':mono,'n_periods':len(monthly),
            'n_samples':len(df),'G1':float(gr.get(1,0)),'G5':float(gr.get(5,0))}

log("="*60)
log("Full Pipeline on RTX 3090")
log("="*60)

# Load embeddings
log("Loading embeddings...")
emb = {}
for c in ['title','summary','full']:
    p = EMBED_DIR / f"embeddings_{c}.npz"
    if p.exists():
        d = np.load(p)
        emb[c] = {k: d[k] for k in ['last_cls','all_cls','all_mean']}
        log(f"  [{c}] all_cls: {d['all_cls'].shape}")
    else: log(f"  [{c}] MISSING")

gap_p = EMBED_DIR / "embeddings_gap.npz"
if gap_p.exists():
    emb['gap'] = {k: np.load(gap_p)[k] for k in np.load(gap_p).files}
    log(f"  [gap] keys={list(emb['gap'].keys())}")

sent_p = EMBED_DIR / "sentiment_finbert.npz"
if sent_p.exists():
    d = np.load(sent_p)
    emb['sentiment'] = {k: d[k] for k in d}
    log(f"  [sentiment] probs: {d['probabilities'].shape}")

# Load labels
log("Loading labels...")
labels = pd.read_csv(DATA_DIR / "reports_with_labels.csv")
n = len(labels)
log(f"Labels: {n:,} rows, columns={list(labels.columns)}")

# Use fwd_excess_5d as main target
target = labels['fwd_excess_5d'].fillna(0).values
dates = labels['tradable_date']
log(f"Target: mean={target.mean():.4f}, std={target.std():.4f}")

# Build factors
log("Building factors...")
factors = {}
n_emb = emb['full']['all_cls'].shape[0]
n_use = min(n, n_emb)

# FHF-CLS: last 4 layers
X = emb['full']['all_cls'][:n_use, -4:, :].reshape(n_use, -1)
factors['FHF-CLS'] = PCA(1, random_state=42).fit_transform(X).flatten()
log(f"  FHF-CLS built")

# FHF-LayerMix: all layers
X = emb['full']['all_cls'][:n_use].reshape(n_use, -1)
factors['FHF-LayerMix'] = PCA(1, random_state=42).fit_transform(X).flatten()

# FHF-TokenPool
X = emb['full']['all_mean'][:n_use].reshape(n_use, -1)
factors['FHF-TokenPool'] = PCA(1, random_state=42).fit_transform(X).flatten()

# FHF-Gap
if 'gap' in emb:
    g1 = emb['gap']['gap_cls'][:n_use].mean(axis=1)
    g2 = emb['gap']['cos_sim'][:n_use]
    X = np.column_stack([g1, g2])
    factors['FHF-Gap'] = PCA(1, random_state=42).fit_transform(X).flatten()

# Baselines
probs = emb['sentiment']['probabilities'][:n_use]
factors['FinBERT-Sentiment'] = probs[:,2] - probs[:,0]
factors['FinBERT-PosProb'] = probs[:,2]
factors['LastCLS-PCA'] = PCA(1, random_state=42).fit_transform(emb['full']['last_cls'][:n_use]).flatten()
factors['Random'] = np.random.RandomState(42).randn(n_use)
factors['ReportCount'] = np.ones(n_use)

log(f"Built {len(factors)} factors")

# Evaluate all
log("\n" + "="*60)
log("FACTOR EVALUATION (CSI300 excess 5d)")
log("="*60)
log(f"{'Factor':22s} {'RankIC':>8s} {'ICIR':>7s} {'t-stat':>7s} {'LS':>8s} {'Mono':>6s} {'Sig':>4s}")
log("-"*62)

results = []
for name in ['FHF-CLS','FHF-LayerMix','FHF-TokenPool','FHF-Gap',
              'FinBERT-Sentiment','FinBERT-PosProb','LastCLS-PCA','Random','ReportCount']:
    if name not in factors: continue
    r = evaluate_factor(factors[name][:n_use], target[:n_use], dates[:n_use], name)
    if r:
        results.append(r)
        sig = '***' if abs(r['IC_t'])>2.58 else ('**' if abs(r['IC_t'])>1.96 else ('*' if abs(r['IC_t'])>1.64 else ''))
        log(f"{name:22s} {r['RankIC']:+8.4f} {r['ICIR']:+7.3f} {r['IC_t']:+7.2f} {r['LS']:+8.4f} {str(r['mono']):>6s} {sig:>4s}")

# Sort and save
results.sort(key=lambda r: abs(r['RankIC']), reverse=True)
with open(RESULTS_DIR / "factor_results.json", 'w') as f:
    json.dump(results, f, indent=2)
log(f"\nBest: {results[0]['name']} (RankIC={results[0]['RankIC']:.4f})")

# FHF vs FinBERT comparison
fhf = [r for r in results if 'FHF' in r['name']]
fin = [r for r in results if r['name']=='FinBERT-Sentiment']
if fhf and fin:
    log(f"FHF-CLS:       IC={fhf[0]['RankIC']:.4f}, t={fhf[0]['IC_t']:.2f}")
    log(f"FinBERT-Sent:  IC={fin[0]['RankIC']:.4f}, t={fin[0]['IC_t']:.2f}")
    log(f"FHF advantage: {abs(fhf[0]['RankIC'])-abs(fin[0]['RankIC']):+.4f}")

log("\nPipeline complete!")
log(f"Results: {RESULTS_DIR}/factor_results.json")
