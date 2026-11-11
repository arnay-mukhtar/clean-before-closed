# POST-REGISTRATION, exploratory (descriptive Q2 only; no outcome data).
# How much to trust the counterfactual behind the 1.5x / 3.6x results.
#  (A) Validation: run the matched counterfactual on units whose future
#      path IS observed and compare prediction with what they emitted:
#      (i) units that retired later, pseudo-dated six years before their
#          last coal year; (ii) units that never exited, given pseudo exit
#          years drawn from the real exit years.
#  (B) Matching variants: 3/5/10 donors, national pool, NOx rate added,
#      3-year horizon.
#  (C) Plant-cluster bootstrap intervals for every valuation multiple.
#  (D) Cumulative deaths avoided through 2024 under each accounting rule.
# Needs us_epa_annual_all.csv, us_unit_exits_damages.csv,
# isrm_marginal_values.csv (all deposited). Runs in about 5-15 minutes.
DIR = '/content/drive/MyDrive/coal_closures'
import os
import numpy as np, pandas as pd

OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
ST2T = 0.90718474
rng = np.random.default_rng(20261109)
NB = 2000

# ---------- data (same construction as outcomes4.py) ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'statecode',
                         'so2mass', 'noxmass', 'heatinput',
                         'primaryfuelinfo'])
for c in ['so2mass', 'noxmass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
coal_uids = set(a.loc[a['coal'], 'uid'])
A = a[a['uid'].isin(coal_uids)].pivot_table(
    index='uid', columns='year', values=['so2mass', 'noxmass', 'heatinput'],
    aggfunc='sum').fillna(0)
YEARS = sorted(A['heatinput'].columns)
for y in range(min(YEARS), max(YEARS) + 1):
    for v in ['so2mass', 'noxmass', 'heatinput']:
        if (v, y) not in A.columns:
            A[(v, y)] = 0.0
A = A.sort_index(axis=1)
H, S2, NX = A['heatinput'], A['so2mass'], A['noxmass']
YMAX = max(YEARS)
state = a.drop_duplicates('uid').set_index('uid')['statecode']
lastcoal = a[a['coal']].groupby('uid')['year'].max()
fid_of = pd.Series({u: int(u.split('_')[0]) for u in A.index})
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)
e10 = a[a['year'] == 2010].groupby('uid')[['so2mass', 'noxmass']].sum()
U = U.join(e10.rename(columns={'so2mass': 'so2_2010', 'noxmass': 'nox_2010'}),
           on='uid')
U[['so2_2010', 'nox_2010']] = U[['so2_2010', 'nox_2010']].fillna(0)
cell_of_fid = U.groupby('fid')['cell'].first()
# coal facilities without an exit: cells from the ISRM facility file
fcc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
cell_all = fcc.set_index('facilityid')['cell'].to_dict()
cell_all.update(cell_of_fid.to_dict())

def mm(cell, kind='MM'):
    c = int(cell)
    if c < 0:
        return np.nan, np.nan
    return (mv[f'{kind}_SO2_high'].values[c] * ST2T,
            mv[f'{kind}_NOx_high'].values[c] * ST2T)

# ---------- matching (as outcomes4.py, with options) ----------
# numpy copies for speed (same numbers as the pandas version in outcomes4)
IDX = list(A.index); POS = {u: i for i, u in enumerate(IDX)}
Y0 = min(YEARS)
Hn, Sn, Nn = H.values, S2.values, NX.values
LC = lastcoal.reindex(IDX).fillna(-1).values
STn = state.reindex(IDX).values
def donors(uid, T, k=5, same_state=True, nox=False, horizon=5):
    i, c = POS[uid], lambda y: y - Y0
    ok = (LC >= T + horizon) & (Hn[:, c(T - 5):c(T - 1) + 1] > 0).all(1)
    ok[i] = False
    pool = np.where(ok)[0]
    if len(pool) < 10:
        return None
    if same_state:
        same = pool[STn[pool] == STn[i]]
        pool = same if len(same) >= k else pool
    h1, h5 = Hn[i, c(T - 1)], Hn[i, c(T - 5)]
    if h1 <= 0 or h5 <= 0:
        return None
    hp1, hp5 = Hn[pool, c(T - 1)], Hn[pool, c(T - 5)]
    x0 = [np.log(h1 / h5), np.log(h1), np.log((Sn[i, c(T - 1)] + 1) / h1)]
    X = [np.log(hp1 / hp5), np.log(hp1),
         np.log((Sn[pool, c(T - 1)] + 1) / hp1)]
    if nox:
        x0.append(np.log((Nn[i, c(T - 1)] + 1) / h1))
        X.append(np.log((Nn[pool, c(T - 1)] + 1) / hp1))
    x0, X = np.array(x0), np.column_stack(X)
    sdv = X.std(0) + 1e-9
    dist = np.sqrt((((X - x0) / sdv) ** 2).sum(1))
    return pool[np.argsort(dist, kind='stable')[:k]]

def ratios(nn, T, horizon=5):
    c = lambda y: y - Y0
    sl = slice(c(T + 1), c(T + horizon) + 1)
    sp, npre = Sn[nn, c(T - 1)].sum(), Nn[nn, c(T - 1)].sum()
    rs = Sn[nn, sl].mean(axis=1).sum() / sp if sp > 0 else 1.0
    rn = Nn[nn, sl].mean(axis=1).sum() / npre if npre > 0 else 1.0
    return rs, rn

def build(k=5, same_state=True, nox=False, horizon=5):
    rows = []
    for u in U.itertuples():
        T = u.last_coal_year
        if u.uid not in POS:
            continue
        if T - 5 < min(YEARS) or T + horizon > YMAX:
            continue
        nn = donors(u.uid, T, k, same_state, nox, horizon)
        if nn is None:
            continue
        rs, rn = ratios(nn, T, horizon)
        ms, mn = mm(u.cell)
        if not np.isfinite(ms):
            continue
        rows.append(dict(uid=u.uid, fid=u.fid, T=T,
                         d_final=u.so2_final * ms + u.nox_final * mn,
                         d_cf=u.so2_final * rs * ms + u.nox_final * rn * mn,
                         d_t10=u.so2_t10 * ms + u.nox_t10 * mn))
    return pd.DataFrame(rows)

# ---------- (A) validation on units with observed futures ----------
print('(A) Validation of the matched counterfactual', flush=True)
real_T = U['last_coal_year'].values
def validate(cands, label):
    rows = []
    for uid, T in cands:
        if T - 5 < min(YEARS) or T + 5 > YMAX or H.at[uid, T - 1] <= 0:
            continue
        nn = donors(uid, T)
        if nn is None:
            continue
        rs, rn = ratios(nn, T)
        ms, mn = mm(cell_all.get(fid_of[uid], -1))
        if not np.isfinite(ms):
            continue
        yrs = list(range(T + 1, T + 6))
        act = S2.loc[uid, yrs].mean() * ms + NX.loc[uid, yrs].mean() * mn
        pre = S2.at[uid, T - 1] * ms + NX.at[uid, T - 1] * mn
        pred = S2.at[uid, T - 1] * rs * ms + NX.at[uid, T - 1] * rn * mn
        rows.append(dict(uid=uid, fid=fid_of[uid], pre=pre, pred=pred,
                         act=act))
    V = pd.DataFrame(rows)
    if len(V) < 10:
        print(f'  {label}: too few units ({len(V)})'); return V
    g = V.groupby('fid')[['pre', 'pred', 'act']].sum()
    def stats(d):
        return d['act'].sum() / d['pred'].sum(), d['pre'].sum() / d['act'].sum(), \
            d['pre'].sum() / d['pred'].sum()
    est = stats(g)
    bs = np.array([stats(g.iloc[rng.integers(0, len(g), len(g))])
                   for _ in range(NB)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f'  {label}: {len(V)} units at {len(g)} plants')
    print(f'    actual / predicted damages, next 5 years: {est[0]:.2f} '
          f'[{lo[0]:.2f}, {hi[0]:.2f}]  (1 = counterfactual unbiased)')
    print(f'    year-before rule / actual: {est[1]:.2f} [{lo[1]:.2f}, '
          f'{hi[1]:.2f}]; year-before rule / predicted: {est[2]:.2f} '
          f'[{lo[2]:.2f}, {hi[2]:.2f}]')
    return V

later = [(u, int(lastcoal[u]) - 6) for u in U['uid']
         if u in lastcoal.index and u in H.index]
V1 = validate(later, '(i) exiting units, pseudo-dated 6 years before exit')
surv = [u for u in lastcoal.index[lastcoal >= YMAX - 1] if u in H.index]
pT = rng.choice(real_T[real_T <= YMAX - 6], len(surv))
V2 = validate(list(zip(surv, pT)), '(ii) units that never exited, pseudo '
              'exit years drawn from real exit years')
pd.concat([V1.assign(group='later_exit'), V2.assign(group='never_exit')]) \
    .to_csv(f'{OUT}/cf_validation.csv', index=False)

# ---------- (B) matching variants ----------
print('\n(B) Matching variants (deaths, InMAP marginal mortality):',
      flush=True)
variants = [('main: 5 donors, same state, 5 years', {}),
            ('3 donors', dict(k=3)), ('10 donors', dict(k=10)),
            ('national pool', dict(same_state=False)),
            ('NOx rate added to matching', dict(nox=True)),
            ('3-year horizon', dict(horizon=3))]
vres = []
for lab, kw in variants:
    C = build(**kw)
    f, cf, t10 = C.d_final.sum(), C.d_cf.sum(), C.d_t10.sum()
    vres.append(dict(variant=lab, exits=len(C), deaths_cf=cf,
                     last_full_year_x=f / cf, ten_years_x=t10 / cf))
    print(f'  {lab}: {len(C)} exits; counterfactual {cf:,.0f} deaths/yr; '
          f'last-full-year {f / cf:.2f}x; ten-years-before {t10 / cf:.2f}x',
          flush=True)
    if not kw:
        MAIN = C
pd.DataFrame(vres).to_csv(f'{OUT}/cf_variants.csv', index=False)

# ---------- (C) bootstrap intervals for the valuation multiples ----------
print('\n(C) Plant-cluster bootstrap, 95% intervals', flush=True)
def unit_vals(kind):
    out = []
    for u in U.itertuples():
        ms, mn = mm(u.cell, kind)
        if not np.isfinite(ms):
            continue
        out.append(dict(fid=u.fid, T=u.last_coal_year,
                        fin=u.so2_final * ms + u.nox_final * mn,
                        t10=u.so2_t10 * ms + u.nox_t10 * mn,
                        peak=u.so2_peak * ms + u.nox_peak * mn,
                        y2010=u.so2_2010 * ms + u.nox_2010 * mn))
    return pd.DataFrame(out)
def boot(df, f):
    g = {k: v for k, v in df.groupby('fid')}
    keys = list(g)
    est = f(df)
    bs = [f(pd.concat([g[keys[i]] for i in rng.integers(0, len(keys),
                                                         len(keys))]))
          for _ in range(NB)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
crow = []
for kind, name in [('MM', 'deaths'), ('MD', 'dollars')]:
    D = unit_vals(kind)
    pos = lambda d, c: d[d[c] > 0]
    tests = {
        'ten years before / last full year':
            lambda d: pos(d, 't10').t10.sum() / pos(d, 't10').fin.sum(),
        'peak / last full year':
            lambda d: pos(d, 'peak').peak.sum() / pos(d, 'peak').fin.sum(),
        '2010 / last full year, exits 2011-2024':
            lambda d: d[d['T'] >= 2011].y2010.sum() / d[d['T'] >= 2011].fin.sum(),
        '2010 / last full year, exits 2020-2024':
            lambda d: d[d['T'] >= 2020].y2010.sum() / d[d['T'] >= 2020].fin.sum()}
    for lab, f in tests.items():
        e, lo, hi = boot(D, f)
        crow.append(dict(weight=name, multiple=lab, est=e, lo=lo, hi=hi))
        print(f'  [{name}] {lab}: {e:.2f} [{lo:.2f}, {hi:.2f}]')
for lab, f in {'last full year / counterfactual':
                   lambda d: d.d_final.sum() / d.d_cf.sum(),
               'ten years before / counterfactual':
                   lambda d: d.d_t10.sum() / d.d_cf.sum()}.items():
    e, lo, hi = boot(MAIN, f)
    crow.append(dict(weight='deaths', multiple=lab, est=e, lo=lo, hi=hi))
    print(f'  [deaths] {lab}: {e:.2f} [{lo:.2f}, {hi:.2f}]  (donor '
          'uncertainty not included)')
pd.DataFrame(crow).to_csv(f'{OUT}/valuation_bootstrap.csv', index=False)

# ---------- (D) cumulative deaths avoided through 2024 ----------
print('\n(D) Cumulative deaths avoided, exits 2011-2024, years after exit '
      'through 2024 (counterfactual ratio held at its 5-year mean):')
M = MAIN.merge(U[['uid', 'so2_2010', 'nox_2010', 'cell']], on='uid')
cum = {'2010 emissions (advocacy rule)': 0.0, 'last full year': 0.0,
       'counterfactual': 0.0}
for r in M[M['T'] >= 2011].itertuples():
    n = 2024 - r.T
    ms, mn = mm(r.cell)
    cum['2010 emissions (advocacy rule)'] += n * (r.so2_2010 * ms
                                                  + r.nox_2010 * mn)
    cum['last full year'] += n * r.d_final
    cum['counterfactual'] += n * r.d_cf
for k, v in cum.items():
    print(f'  {k}: {v:,.0f} deaths '
          f'({v / cum["counterfactual"]:.1f}x the counterfactual)')
print(f'  ({(M["T"] >= 2011).sum()} matched exits; unmatched exits omitted)')
pd.Series(cum).to_csv(f'{OUT}/cumulative_deaths.csv')
