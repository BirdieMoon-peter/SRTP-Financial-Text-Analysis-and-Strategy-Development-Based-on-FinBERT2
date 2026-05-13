"""
SRTP Enhanced Experiments — Optimized for speed
=================================================
Fixed n_components (no expensive CV search), representative layers only.
Expected runtime: ~40 min on RTX 3090.

Key experiments:
  1. Per-layer PLS (layers 1, 4, 8, 12 — representative)
  2. Ridge ensemble vs IC-weighted vs equal-weight
  3. Gap per-layer analysis
  4. Rolling window optimization
  5. Layer-weighted Ridge ensemble
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Paths
EMBED = None; DATA = None; RES = None
for base in [Path("/root/srtp"), Path("/root/autodl-tmp/srtp"),
             Path(__file__).resolve().parent.parent]:
    if (base / "data" / "reports_with_labels.csv").exists():
        PROJ = base; DATA = PROJ / "data"
        for emb_base in [Path("/root/autodl-tmp/srtp/embeddings"),
                         DATA / "embeddings", PROJ / "embeddings"]:
            if (emb_base / "embeddings_full.npz").exists():
                EMBED = emb_base; break
        RES = PROJ / "results"; break
if DATA is None or EMBED is None:
    raise FileNotFoundError("Cannot find project root or embeddings")
RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def ws(x, lo=0.01, hi=0.99):
    x=np.asarray(x,float); v=~np.isnan(x)
    return np.clip(x,*np.nanquantile(x[v],[lo,hi])) if v.sum()>0 else x
def stdz(x):
    v=~np.isnan(x); r=np.full_like(x,np.nan,dtype=float)
    if v.sum()==0: return r
    r[v]=(x[v]-x[v].mean())/x[v].std(); return r

def eval_factor(name, factor, ret, dates):
    df=pd.DataFrame({'f':np.asarray(factor,float),'r':np.asarray(ret,float),
                     'd':pd.to_datetime(dates)}).dropna()
    if len(df)<50: return None
    m_ics=[stats.spearmanr(g['f'],g['r'])[0]
           for _,g in df.groupby(df['d'].dt.to_period('M')) if len(g)>=10]
    if len(m_ics)<6: return None
    im,sd=np.mean(m_ics),np.std(m_ics); T=len(m_ics); t=im/sd*np.sqrt(T) if sd>0 else 0
    try: df['g']=pd.qcut(df['f'],5,labels=False,duplicates='drop')+1
    except: df['g']=1
    gr={g:df[df['g']==g]['r'].mean() for g in range(1,6)}
    return {'name':name,'RankIC':float(im),'ICIR':float(im/sd) if sd>0 else 0,
            'IC_t':float(t),'LS':float(gr.get(5,np.nan)-gr.get(1,np.nan)),
            'G1':float(gr.get(1,0)),'G5':float(gr.get(5,0)),
            'hit_ratio':float(np.mean([ic>0 for ic in m_ics])),'n_periods':T,'n':len(df)}

def rolling_pls_fast(X, y, dates, n_components=5, train_window=24):
    """Rolling PLS factor construction — fixed n_components, no CV overhead."""
    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    periods = dates_idx.to_period('M')
    months = sorted(periods.unique())
    factors = np.full(len(X), np.nan)
    count=0
    for month in months:
        ts = month - pd.offsets.MonthEnd(train_window)
        tm = (periods >= ts) & (periods < month)
        xm = periods == month
        if tm.sum() < 100 or xm.sum() < 10: continue
        Xt_raw, yt_raw = X[tm], y[tm]
        valid = ~np.isnan(yt_raw)
        Xt, yt = Xt_raw[valid], yt_raw[valid]
        if len(Xt) < 100: continue
        nc = min(n_components, Xt.shape[1]//100, Xt.shape[0]//50)
        nc = max(1, nc)
        try:
            sc = StandardScaler()
            Xt_s = sc.fit_transform(Xt)
            Xe_s = sc.transform(X[xm])
            pls = PLSRegression(n_components=nc, scale=False)
            pls.fit(Xt_s, yt.reshape(-1,1))
            r = pls.transform(Xe_s)
            if isinstance(r, tuple): r = r[0]
            factors[xm] = r[:, 0]
            count+=1
        except Exception: continue
    log(f"  [PLS] pred={int((~np.isnan(factors)).sum())}/{len(X)} months={count} nc={n_components}")
    return factors

# ============================================================
def exp1_per_layer(full_emb, labels, dates, target):
    """Per-layer PLS: test representative layers (1,4,8,12)."""
    log("\n"+"="*60); log("EXP1: Per-Layer PLS (layers 1,4,8,12)"); log("="*60)
    all_cls = full_emb['all_cls']  # (N, 12, 768)
    results=[]
    for layer in [0, 3, 7, 11]:  # layers 1, 4, 8, 12 (0-indexed)
        X = all_cls[:, layer, :]
        log(f"  Layer {layer+1}: X={X.shape}")
        # PCA
        f_pca = rolling_pls_fast(X, target, dates, n_components=1, train_window=24)
        r = eval_factor(f"L{layer+1:02d}-PCA", f_pca, target, dates)
        if r: results.append(r); log(f"    PCA: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
        # PLS
        f_pls = rolling_pls_fast(X, target, dates, n_components=5, train_window=24)
        r = eval_factor(f"L{layer+1:02d}-PLS", f_pls, target, dates)
        if r: results.append(r); log(f"    PLS: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    return results

def exp2_ridge_ensemble(full_emb, labels, dates, target, sent_score):
    """Compare ensemble methods: equal, IC-weighted, Ridge."""
    log("\n"+"="*60); log("EXP2: Ensemble Methods"); log("="*60)
    all_cls = full_emb['all_cls']
    # Build base factors
    X_last4 = all_cls[:, -4:, :].reshape(all_cls.shape[0], -1)
    X_all12 = all_cls.reshape(all_cls.shape[0], -1)
    f1 = stdz(rolling_pls_fast(X_last4, target, dates, nc=5))
    f2 = stdz(rolling_pls_fast(X_all12, target, dates, nc=5))
    f3 = stdz(sent_score)
    factors = {'CLS-PLS':f1, 'AllCLS-PLS':f2, 'Sentiment':f3}
    F = np.column_stack([v for v in factors.values()])
    log(f"  Factor matrix: {F.shape}")

    # Equal weight
    eq = stdz(F.mean(axis=1))
    results=[eval_factor("Ens-Equal", eq, target, dates)]

    # IC-weighted (rolling)
    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    periods = dates_idx.to_period('M')
    months = sorted(periods.unique())
    icw = np.full(F.shape[0], np.nan)
    for month in months:
        ts = month - pd.offsets.MonthEnd(24)
        tm = (periods >= ts) & (periods < month)
        xm = periods == month
        if tm.sum()<50 or xm.sum()<10: continue
        F_tr, y_tr = F[tm], target[tm]
        v = ~np.isnan(y_tr) & ~np.isnan(F_tr).any(axis=1)
        F_tr, y_tr = F_tr[v], y_tr[v]
        if len(F_tr)<50: continue
        ics = [abs(stats.spearmanr(F_tr[:,k], y_tr)[0]) for k in range(F.shape[1])]
        ics = [max(0, ic if not np.isnan(ic) else 0) for ic in ics]
        if sum(ics)>0:
            w = np.array(ics)/sum(ics)
            icw[xm] = F[xm] @ w
    icw = stdz(icw)
    r = eval_factor("Ens-ICW", icw, target, dates)
    if r: results.append(r); log(f"  ICW: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

    # Ridge ensemble (rolling)
    ridge_f = np.full(F.shape[0], np.nan)
    for month in months:
        ts = month - pd.offsets.MonthEnd(24)
        tm = (periods >= ts) & (periods < month)
        xm = periods == month
        if tm.sum()<100 or xm.sum()<10: continue
        F_tr, y_tr = F[tm], target[tm]
        v = ~np.isnan(y_tr) & ~np.isnan(F_tr).any(axis=1)
        F_tr, y_tr = F_tr[v], y_tr[v]
        if len(F_tr)<100: continue
        try:
            ridge = RidgeCV(alphas=np.logspace(-1,2,10))
            ridge.fit(F_tr, y_tr)
            ridge_f[xm] = ridge.predict(F[xm])
        except: continue
    ridge_f = stdz(ridge_f)
    r = eval_factor("Ens-Ridge", ridge_f, target, dates)
    if r: results.append(r); log(f"  Ridge: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    return results

def exp3_gap_enhanced(gap_emb, target, dates):
    """Per-layer gap analysis."""
    log("\n"+"="*60); log("EXP3: Per-Layer Gap"); log("="*60)
    cos = gap_emb['cos_sim']  # (N, 12)
    n_layers = cos.shape[1]
    results=[]
    for layer in [0, 3, 7, 11]:
        gap_l = ws(1.0 - cos[:, layer])
        r = eval_factor(f"Gap-L{layer+1:02d}", gap_l, target, dates)
        if r: results.append(r); log(f"  L{layer+1}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    # Best 3 layers combined via PLS
    top_gaps = np.column_stack([1.0-cos[:,l] for l in [0,3,7,11]])
    f_pls = rolling_pls_fast(top_gaps, target, dates, nc=3)
    r = eval_factor("Gap-4Layer-PLS", f_pls, target, dates)
    if r: results.append(r); log(f"  4Layer-PLS: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    return results

def exp4_window_optimization(full_emb, target, dates):
    """Test rolling windows 12/18/24/30/36 months."""
    log("\n"+"="*60); log("EXP4: Window Optimization"); log("="*60)
    X = full_emb['all_cls'][:, -4:, :].reshape(full_emb['all_cls'].shape[0], -1)
    results=[]
    for w in [12, 18, 24, 30, 36]:
        f = rolling_pls_fast(X, target, dates, n_components=5, train_window=w)
        r = eval_factor(f"Win-{w:02d}M", f, target, dates)
        if r: results.append(r); log(f"  Win{w:2d}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    return results

def exp5_layer_weighted(full_emb, target, dates):
    """Layer-weighted Ridge ensemble on per-layer PLS factors."""
    log("\n"+"="*60); log("EXP5: Layer-Weighted Ensemble"); log("="*60)
    all_cls = full_emb['all_cls']
    n_layers=all_cls.shape[1]
    dates_idx=pd.DatetimeIndex(pd.to_datetime(dates))
    periods=dates_idx.to_period('M')
    months=sorted(periods.unique())

    # Build per-layer PLS factors
    per_layer = np.full((all_cls.shape[0], n_layers), np.nan)
    for l in range(n_layers):
        X=all_cls[:,l,:]
        per_layer[:,l] = rolling_pls_fast(X, target, dates, nc=3)
    for l in range(n_layers): per_layer[:,l] = stdz(per_layer[:,l])

    # Ridge combine
    ridge_f=np.full(all_cls.shape[0], np.nan)
    for month in months:
        ts=month-pd.offsets.MonthEnd(24)
        tm=(periods>=ts)&(periods<month); xm=periods==month
        if tm.sum()<100 or xm.sum()<10: continue
        Ft,yt=per_layer[tm],target[tm]
        v=~np.isnan(yt)&~np.isnan(Ft).any(axis=1)
        Ft,yt=Ft[v],yt[v]
        if len(Ft)<100: continue
        try:
            ridge=RidgeCV(alphas=np.logspace(-1,2,10))
            ridge.fit(Ft,yt)
            ridge_f[xm]=ridge.predict(per_layer[xm])
        except: continue
    ridge_f=stdz(ridge_f)
    mean_f=stdz(np.nanmean(per_layer,axis=1))
    results=[]
    for name,f in [("LayerWeight-Ridge",ridge_f),("LayerWeight-Mean",mean_f)]:
        r=eval_factor(name,f,target,dates)
        if r: results.append(r); log(f"  {name}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")
    return results

# ============================================================
def main():
    log("="*60); log("SRTP Enhanced Experiments v2"); log("="*60)
    log("Loading data...")
    labels=pd.read_csv(DATA/"reports_with_labels.csv")
    labels['report_date']=pd.to_datetime(labels['report_date'])
    labels['tradable_date']=pd.to_datetime(labels['tradable_date'])
    full=np.load(EMBED/"embeddings_full.npz")
    gap=np.load(EMBED/"embeddings_gap.npz")
    sent=np.load(EMBED/"sentiment_finbert.npz")['probabilities']

    n=min(len(labels), full['all_cls'].shape[0])
    log(f"  Samples: {n:,}")
    labels=labels.iloc[:n]
    target5=ws(labels['fwd_excess_5d'].fillna(0).values[:n])
    dates=labels['tradable_date'].values[:n]
    sent_score=sent[:n,2]-sent[:n,0]

    # Truncate embeddings
    full_t={k:full[k][:n] for k in full.files}
    gap_t={k:gap[k][:n] for k in gap.files}

    r=eval_factor("Baseline-Sentiment", sent_score, target5, dates)
    log(f"\n  Baseline: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}" if r else "\n  Baseline: ERROR")

    all_results=[]
    for fn,name in [(exp1_per_layer,"EXP1"),(exp2_ridge_ensemble,"EXP2"),
                     (exp3_gap_enhanced,"EXP3"),(exp4_window_optimization,"EXP4"),
                     (exp5_layer_weighted,"EXP5")]:
        try:
            log(f"\n{'='*40}"); log(f"Running {name}...")
            if name=="EXP1": results=fn(full_t, labels, dates, target5)
            elif name=="EXP2": results=fn(full_t, labels, dates, target5, sent_score)
            elif name=="EXP3": results=fn(gap_t, target5, dates)
            elif name in ("EXP4","EXP5"): results=fn(full_t, target5, dates)
            all_results.extend(results or [])
        except Exception as e:
            log(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()

    # Save
    log("\n"+"="*60); log("RESULTS"); log("="*60)
    all_results.sort(key=lambda x: abs(x['RankIC']), reverse=True)
    for r in all_results[:20]:
        sig="***" if abs(r['IC_t'])>2.58 else ("**" if abs(r['IC_t'])>1.96 else "")
        log(f"  {r['name']:<25s} IC={r['RankIC']:>+8.4f}  ICIR={r['ICIR']:>7.3f}  t={r['IC_t']:>+7.2f}{sig}  hit={r['hit_ratio']:.1%}")

    with open(RES/"enhanced_experiments_v2.json",'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    pd.DataFrame(all_results).to_csv(RES/"enhanced_experiments_v2.csv", index=False)
    log(f"\nSaved to {RES/'enhanced_experiments_v2.json'}"); log("Done!")

if __name__=="__main__":
    main()
