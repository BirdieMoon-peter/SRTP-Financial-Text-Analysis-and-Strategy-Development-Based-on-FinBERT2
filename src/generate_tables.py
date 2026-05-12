"""
Generate all remaining tables for the thesis.
Runs on Windows server with existing data.
"""
import json, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from collections import OrderedDict, defaultdict
warnings.filterwarnings("ignore")

DATA = Path("C:/Users/13082/CSMAR/data")
EMBED = DATA / "embeddings"
RES = Path("C:/Users/13082/CSMAR/results")
RES.mkdir(exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def ws(x, lo=0.01, hi=0.99):
    x=np.asarray(x,float); v=~np.isnan(x)
    return np.clip(x,*np.nanquantile(x[v],[lo,hi])) if v.sum()>0 else x
def rank_ic(factor, ret, dates):
    df=pd.DataFrame({'f':np.asarray(factor,float),'r':np.asarray(ret,float),
                     'd':pd.to_datetime(dates)}).dropna()
    if len(df)<50: return None
    ics=[stats.spearmanr(g['f'],g['r'])[0] for _,g in df.groupby(df['d'].dt.to_period('M')) if len(g)>=10]
    if not ics: return None
    m,s=np.mean(ics),np.std(ics)
    return {'RankIC':float(m),'ICIR':float(m/s) if s>0 else 0,
            'IC_t':float(m/s*np.sqrt(len(ics))) if s>0 else 0,'n_periods':len(ics)}

log("="*60)
log("GENERATING ALL THESIS TABLES")
log("="*60)

log("Loading data...")
labels=pd.read_csv(DATA/"reports_with_labels.csv")
labels['report_date']=pd.to_datetime(labels['report_date'])
labels['tradable_date']=pd.to_datetime(labels['tradable_date'])

sent=np.load(EMBED/"sentiment_finbert.npz")['probabilities']
gap_emb=np.load(EMBED/"embeddings_gap.npz")
full_emb=np.load(EMBED/"embeddings_full.npz")

n=min(len(labels),full_emb['all_cls'].shape[0])
log(f"n={n:,}")

labels=labels.iloc[:n]; dates=labels['tradable_date']
codes=labels['stock_code'].astype(str).str.zfill(6)
labels['stock_code']=codes
sent_score=sent[:n,2]-sent[:n,0]
target5=ws(labels['fwd_excess_5d'].fillna(0).values[:n])

# Stock data
stock=pd.read_csv(DATA/"csmar_daily_stock.csv",dtype={"stock_code":str},low_memory=False)
stock['date']=pd.to_datetime(stock['date'])
stock['stock_code']=stock['stock_code'].astype(str).str.zfill(6)
stock=stock.sort_values(['stock_code','date']).reset_index(drop=True)
for c in ['close','volume','market_cap']:
    if c in stock.columns: stock[c]=pd.to_numeric(stock[c],errors='coerce')

stock['ret_5d']=stock.groupby('stock_code')['close'].pct_change(5)
stock['ret_20d']=stock.groupby('stock_code')['close'].pct_change(20)
stock['ret_60d']=stock.groupby('stock_code')['close'].pct_change(60)
stock['vol_20d']=stock.groupby('stock_code')['close'].pct_change().transform(lambda x:x.rolling(20).std())
stock['turnover']=stock['volume']/stock.groupby('stock_code')['volume'].transform(lambda x:x.rolling(60).mean().clip(lower=1))
stock['log_vol']=np.log1p(stock['volume'])
stock['log_mktcap']=np.log(stock['amount'].clip(lower=1e6)) if 'amount' in stock.columns else np.zeros(len(stock))

ctrl=stock[['stock_code','date','ret_5d','ret_20d','ret_60d','vol_20d','turnover','log_vol','log_mktcap']]
ctrl=ctrl.rename(columns={'date':'tradable_date'})
ctrl['tradable_date']=pd.to_datetime(ctrl['tradable_date'])
ctrl['stock_code']=ctrl['stock_code'].astype(str).str.zfill(6)
lm=labels.merge(ctrl,on=['stock_code','tradable_date'],how='left')

ind_df=pd.read_csv(DATA/"industry_mapping.csv",dtype={"stock_code":str})
ind_df['stock_code']=ind_df['stock_code'].astype(str).str.zfill(6)
ind_map=dict(zip(ind_df['stock_code'],ind_df['industry_code']))
ind_name_map=dict(zip(ind_df['stock_code'],ind_df['industry_name']))
lm['ind_code']=lm['stock_code'].map(ind_map)
lm['ind_name']=lm['stock_code'].map(ind_name_map)
ind_dummies=pd.get_dummies(lm['ind_code'],prefix='ind',dummy_na=True).astype(float)

rc=lm.groupby('stock_code')['report_date'].transform('count').values[:n]
cov_log=np.log1p(rc.astype(float))

# Targets
targets={}
for c in ['fwd_excess_1d','fwd_excess_2d','fwd_excess_5d','fwd_excess_10d','fwd_excess_20d']:
    if c in labels.columns: targets[c]=ws(labels[c].fillna(0).values[:n])

# ============================================================
# TABLE: Decile Portfolio Analysis
# ============================================================
log("\n--- Decile Portfolio ---")
df=pd.DataFrame({'f':sent_score,'r':target5,'d':pd.to_datetime(dates)}).dropna()
monthly=df.groupby(df['d'].dt.to_period('M'))

groups={i:[] for i in range(1,11)}
ls_vals=[]
for p,g in monthly:
    if len(g)<50: continue
    try:
        g=g.copy(); g['g']=pd.qcut(g['f'],10,labels=False,duplicates='drop')+1
    except: continue
    for i in range(1,11):
        gg=g[g['g']==i]; groups[i].append(gg['r'].mean() if len(gg)>0 else np.nan)
    ls_vals.append(groups[10][-1]-groups[1][-1])

decile={}
for i in range(1,11):
    r=np.array(groups[i]); r=r[~np.isnan(r)]
    decile[f'D{i}']={'mean':float(r.mean()),'hit':float((r>0).mean())}

ls=np.array(ls_vals); ls=ls[~np.isnan(ls)]
decile['LS']={'mean':float(ls.mean()),'t':float(ls.mean()/ls.std()*np.sqrt(len(ls))) if ls.std()>0 else 0,
              'hit':float((ls>0).mean()),'n':len(ls)}
means=[decile[f'D{i}']['mean'] for i in range(1,11)]
decile['monotonic']=all(m1<=m2 for m1,m2 in zip(means,means[1:]) if not(np.isnan(m1) or np.isnan(m2)))

log(f"  LS={decile['LS']['mean']:+.5f} t={decile['LS']['t']:+.2f} monotonic={decile['monotonic']}")

with open(RES/"decile_analysis.json",'w',encoding='utf-8') as f:
    json.dump(decile,f,indent=2)

# ============================================================
# TABLE: Incremental Fama-MacBeth
# ============================================================
log("\n--- Incremental FM ---")
ctrl_dict={}
for c in ['ret_5d','ret_20d','ret_60d','vol_20d','turnover','log_vol','log_mktcap']:
    if c in lm.columns: ctrl_dict[c]=ws(lm[c].fillna(0).values[:n])
ctrl_dict['report_count_log']=cov_log

specs=OrderedDict([
    ('Univariate',[]),
    ('+Coverage',['report_count_log']),
    ('+Momentum',['ret_5d','ret_20d']),
    ('+Liquidity',['turnover','log_vol','vol_20d']),
    ('+Size',['log_mktcap']),
    ('+All_Traditional',['ret_5d','ret_20d','ret_60d','turnover','log_vol','vol_20d','log_mktcap']),
    ('+All+Coverage',['report_count_log','ret_5d','ret_20d','ret_60d','turnover','log_vol','vol_20d','log_mktcap']),
])

fm_df=pd.DataFrame({'f':sent_score,'r':target5,'d':pd.to_datetime(dates)})
for k,v in ctrl_dict.items(): fm_df[k]=np.asarray(v)

fm_results=OrderedDict()
for spec_name,ctrl_list in specs.items():
    cols=['f']+[c for c in ctrl_list if c in fm_df.columns]
    coefs=[]; r2s=[]
    for p,g in fm_df.groupby(fm_df['d'].dt.to_period('M')):
        gs=g[cols+['r']].dropna()
        if len(gs)<20: continue
        try:
            lr=LinearRegression().fit(gs[cols].values,gs['r'].values)
            coefs.append(lr.coef_[0]); r2s.append(lr.score(gs[cols].values,gs['r'].values))
        except: continue
    if not coefs: continue
    cf=np.array(coefs); T=len(cf); m,s=cf.mean(),cf.std()
    fm_results[spec_name]={'fm_coef':float(m),'fm_t':float(m/(s/np.sqrt(T))) if s>0 else 0,
                            'avg_r2':float(np.mean(r2s)),'n_periods':T}
    log(f"  FM [{spec_name:20s}]: coef={m:+.6f} t={fm_results[spec_name]['fm_t']:+.2f} R2={np.mean(r2s):.4f}")

with open(RES/"incremental_fm.json",'w',encoding='utf-8') as f:
    json.dump(fm_results,f,indent=2)

# ============================================================
# TABLE: Orthogonalization
# ============================================================
log("\n--- Orthogonalization ---")
def orth(factor,mat):
    f=np.asarray(factor,float); X=np.asarray(mat,float)
    v=~np.isnan(f)&~np.isnan(X).any(axis=1)
    if v.sum()<50: return f
    lr=LinearRegression().fit(X[v],f[v])
    res=f.copy(); res[v]=f[v]-lr.predict(X[v])
    return res

ind_mat=ind_dummies.values.astype(float)[:n]
mkt=np.asarray(ctrl_dict.get('log_mktcap',np.zeros(n))).reshape(-1,1)
mom=np.asarray(ctrl_dict.get('ret_20d',np.zeros(n))).reshape(-1,1)
rev=np.asarray(ctrl_dict.get('ret_5d',np.zeros(n))).reshape(-1,1)
turn=np.asarray(ctrl_dict.get('turnover',np.zeros(n))).reshape(-1,1)
vol=np.asarray(ctrl_dict.get('vol_20d',np.zeros(n))).reshape(-1,1)

orth_mats=OrderedDict([
    ('Raw',None),
    ('+Industry',ind_mat),
    ('+Industry+Size',np.column_stack([ind_mat,mkt])),
    ('+Ind+Size+Mom',np.column_stack([ind_mat,mkt,mom,rev])),
    ('+Ind+Size+Mom+Turn',np.column_stack([ind_mat,mkt,mom,rev,turn,vol])),
])

orth_res=OrderedDict()
for level,mat in orth_mats.items():
    f_orth=sent_score if mat is None else orth(sent_score,mat)
    r=rank_ic(f_orth,target5,dates)
    if r:
        orth_res[level]={'RankIC':r['RankIC'],'IC_t':r['IC_t'],'ICIR':r['ICIR']}
        log(f"  {level:25s}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

with open(RES/"orthogonalization.json",'w',encoding='utf-8') as f:
    json.dump(orth_res,f,indent=2)

# ============================================================
# TABLE: Robustness Windows
# ============================================================
log("\n--- Robustness Windows ---")
win_r=OrderedDict()
for col in ['fwd_excess_1d','fwd_excess_2d','fwd_excess_5d','fwd_excess_10d','fwd_excess_20d']:
    if col in targets:
        r=rank_ic(sent_score,targets[col],dates)
        if r: win_r[col]=r; log(f"  {col}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f}")

with open(RES/"robustness_windows.json",'w',encoding='utf-8') as f:
    json.dump(win_r,f,indent=2)

# ============================================================
# TABLE: Sub-Periods
# ============================================================
log("\n--- Sub-Periods ---")
periods_dict=OrderedDict([
    ('2020-2021',('2020-01-01','2021-12-31')),
    ('2022',('2022-01-01','2022-12-31')),
    ('2023-2024',('2023-01-01','2024-12-31')),
    ('2025-2026',('2025-01-01','2026-12-31')),
])
per_r=OrderedDict()
for pn,(s,e) in periods_dict.items():
    mask=(pd.to_datetime(dates)>=s)&(pd.to_datetime(dates)<=e)
    if mask.sum()<100: continue
    r=rank_ic(sent_score[mask.values],target5[mask.values],pd.to_datetime(dates)[mask.values])
    if r:
        per_r[pn]={k:v for k,v in r.items() if k!='ics_series'}
        per_r[pn]['n']=int(mask.sum())
        log(f"  {pn}: IC={r['RankIC']:+.4f} t={r['IC_t']:+.2f} n={mask.sum()}")

with open(RES/"robustness_periods.json",'w',encoding='utf-8') as f:
    json.dump(per_r,f,indent=2)

# ============================================================
# TABLE: Size/Coverage/Industry Groups
# ============================================================
log("\n--- Groups ---")
grp=OrderedDict()

# Size
mktv=np.asarray(ctrl_dict.get('log_mktcap',np.zeros(n)))
vm=~np.isnan(mktv)
if vm.sum()>100:
    sl=pd.qcut(pd.Series(mktv[vm]),3,labels=['Small','Mid','Large'],duplicates='drop')
    ss=pd.Series(index=range(n),dtype=object)
    vi=np.where(vm)[0]
    for ii,idx in enumerate(vi):
        if ii<len(sl): ss.iloc[idx]=sl.iloc[ii]
    grp['size']=OrderedDict()
    for lab in ['Small','Mid','Large']:
        mask=ss==lab
        if mask.sum()<100: continue
        r=rank_ic(sent_score[mask.values],target5[mask.values],pd.to_datetime(dates)[mask.values])
        if r: grp['size'][str(lab)]={k:v for k,v in r.items() if k!='ics_series'}; grp['size'][str(lab)]['n']=int(mask.sum())

# Coverage
vc=~np.isnan(cov_log)
if vc.sum()>100:
    cl=pd.qcut(pd.Series(cov_log[vc]),3,labels=['Low','Mid','High'],duplicates='drop')
    cs2=pd.Series(index=range(n),dtype=object)
    vi=np.where(vc)[0]
    for ii,idx in enumerate(vi):
        if ii<len(cl): cs2.iloc[idx]=cl.iloc[ii]
    grp['coverage']=OrderedDict()
    for lab in ['Low','Mid','High']:
        mask=cs2==lab
        if mask.sum()<100: continue
        r=rank_ic(sent_score[mask.values],target5[mask.values],pd.to_datetime(dates)[mask.values])
        if r: grp['coverage'][str(lab)]={k:v for k,v in r.items() if k!='ics_series'}; grp['coverage'][str(lab)]['n']=int(mask.sum())

log(f"  Size groups: {list(grp.get('size',{}).keys())}")
log(f"  Coverage groups: {list(grp.get('coverage',{}).keys())}")

with open(RES/"robustness_groups.json",'w',encoding='utf-8') as f:
    json.dump(grp,f,indent=2,default=str)

# ============================================================
# TABLE: Placebo
# ============================================================
log("\n--- Placebo (500x) ---")
np.random.seed(42)
N=500
placebo={'shuffled_ret':[],'random_match':[]}
for i in range(N):
    sr=target5.copy(); np.random.shuffle(sr)
    r=rank_ic(sent_score,sr,dates)
    if r: placebo['shuffled_ret'].append(r['RankIC'])
    pi=np.random.permutation(n)
    r=rank_ic(sent_score,target5[pi],dates)
    if r: placebo['random_match'].append(r['RankIC'])
    if (i+1)%100==0: log(f"  {i+1}/{N}")

re_ics=[]
for i in range(50):
    r=rank_ic(np.random.randn(n),target5,dates)
    if r: re_ics.append(r['RankIC'])
placebo['random_emb']=re_ics

ps=OrderedDict()
real_ic=0.0154
for k,arr in placebo.items():
    arr=np.array(arr)
    rp=float((arr<real_ic).mean())
    ps[k]={'mean':float(arr.mean()),'std':float(arr.std()),
           'p95':float(np.percentile(arr,95)),'p99':float(np.percentile(arr,99)),
           'real_pct':rp,'n':len(arr)}
    log(f"  {k}: mean={arr.mean():+.4f} p95={np.percentile(arr,95):+.4f} real@{rp:.1%}")

with open(RES/"placebo_results.json",'w',encoding='utf-8') as f:
    json.dump(ps,f,indent=2)

# ============================================================
# TABLE: Constrained Backtest
# ============================================================
log("\n--- Constrained Backtest ---")
def cbacktest(factor,ret,dates,codes,ind_codes,mktcap,top_n=50,tc=0.003):
    df=pd.DataFrame({'f':np.asarray(factor,float),'r':np.asarray(ret,float),
                     'd':pd.to_datetime(dates),'code':codes,'ind':ind_codes,
                     'mkt':np.asarray(mktcap,float)}).dropna()
    periods=sorted(df['d'].dt.to_period('M').unique())
    pr=[]; br=[]; prev=set()
    for p in periods:
        g=df[df['d'].dt.to_period('M')==p]
        if len(g)<top_n*2: continue
        ranked=g.sort_values('f',ascending=False)
        sel=[]; iw=defaultdict(float)
        for _,row in ranked.iterrows():
            if iw.get(row['ind'],0)<0.10:
                if len(sel)<top_n: sel.append(row.name); iw[row['ind']]+=1/top_n
        if len(sel)<top_n//2: continue
        sl=ranked.loc[sel]
        cur=set(sl['code'])
        to=1-len(prev&cur)/max(len(cur),1) if prev else 1
        pr.append(sl['r'].mean()-to*tc); br.append(g['r'].mean()); prev=cur
    if len(pr)<6: return None
    pr=np.array(pr); br=np.array(br); ex=pr-br
    cum=np.cumprod(1+pr)
    return {'ann_return':float((1+pr.mean())**12-1),'ann_vol':float(pr.std()*np.sqrt(12)),
            'sharpe':float((1+pr.mean())**12-1)/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0,
            'max_dd':float(max((np.maximum.accumulate(cum)-cum)/np.maximum.accumulate(cum))),
            'info_ratio':float(ex.mean()/ex.std()*np.sqrt(12)) if ex.std()>0 else 0,
            'win_rate':float((pr>0).mean()),'n_months':len(pr),'cum_return':float(cum[-1]-1)}

bt_all={}
for tc in [0.001,0.002,0.003,0.005]:
    bt=cbacktest(sent_score,target5,dates,codes,lm['ind_code'].values[:n],
                  np.asarray(ctrl_dict.get('log_mktcap',np.zeros(n))),tc=tc)
    if bt: bt_all[f'tc_{int(tc*10000)}bps']=bt
    log(f"  TC={tc:.1%}: AnnRet={bt['ann_return']:+.2%} IR={bt['info_ratio']:.2f}")

with open(RES/"constrained_backtest.json",'w',encoding='utf-8') as f:
    json.dump(bt_all,f,indent=2)

# ============================================================
log("\n"+"="*60)
log("ALL TABLES GENERATED")
log("="*60)
for fn in sorted(os.listdir(str(RES))):
    log(f"  {fn}")
