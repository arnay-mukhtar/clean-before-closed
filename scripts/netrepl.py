# POST-REGISTRATION, exploratory (descriptive Q2 extension; no outcome data).
# How much of a coal exit's emissions cut comes back at other plants that
# replace its output? Panel of regions x years, 1998-2025:
#   dOther[r,t] = year FE (+ region FE) + b0 dExit[r,t] + b1 dExit[r,t-1]
#                 + lead dExit[r,t+1]
# dExit = year-on-year change of units in their exit year or the year after
# (the retirement step); dOther = change of every other unit in the region,
# any fuel, including new units. The offset share is -(b0 + b1): 0 means
# nothing replaced in the region, 1 means fully replaced at the same
# emission rate. The lead is a placebo (should be about 0).
# Regions: state (48 clusters) and census division (9; wild bootstrap with
# Webb weights). Outcomes: heat input, gross load, SO2, NOx and deaths
# (InMAP marginal mortality; coal plants at their own cell, other plants at
# the state mean of coal-plant cells).
DIR = '/content/drive/MyDrive/coal_closures'
import os
import numpy as np, pandas as pd

OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
ST2T = 0.90718474
rng = np.random.default_rng(20261109)

cols = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', nrows=0).columns
use = ['facilityid', 'unitid', 'year', 'statecode', 'so2mass', 'noxmass',
       'heatinput'] + (['grossload'] if 'grossload' in cols else [])
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False, usecols=use)
VARS = [v for v in ['heatinput', 'grossload', 'so2mass', 'noxmass'] if v in a]
for c in VARS:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
L = U.set_index('uid')['last_coal_year']
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
fc = fc[fc['cell'] >= 0]
fc['ms'] = mv['MM_SO2_high'].values[fc['cell'].astype(int)] * ST2T
fc['mn'] = mv['MM_NOx_high'].values[fc['cell'].astype(int)] * ST2T
fst = a.drop_duplicates('facilityid').set_index('facilityid')['statecode']
fc['state'] = fc['facilityid'].map(fst)
st_ms = fc.groupby('state')['ms'].mean(); st_mn = fc.groupby('state')['mn'].mean()
own = fc.set_index('facilityid')[['ms', 'mn']]
a['ms'] = a['facilityid'].map(own['ms']).fillna(a['statecode'].map(st_ms)) \
    .fillna(fc['ms'].mean())
a['mn'] = a['facilityid'].map(own['mn']).fillna(a['statecode'].map(st_mn)) \
    .fillna(fc['mn'].mean())
a['deaths'] = a['so2mass'] * a['ms'] + a['noxmass'] * a['mn']
VARS.append('deaths')

# unit panel with explicit zeros, then year-on-year changes
P = a.pivot_table(index='uid', columns='year', values=VARS, aggfunc='sum') \
    .fillna(0)
yrs = sorted(set(P.columns.get_level_values(1)))
ust = a.drop_duplicates('uid').set_index('uid')['statecode']
rows = []
for v in VARS:
    M = P[v].reindex(columns=yrs, fill_value=0)
    D = M.diff(axis=1).iloc[:, 1:]
    D = D.stack().rename('d').reset_index()
    D.columns = ['uid', 'year', 'd']
    D['var'] = v
    rows.append(D)
D = pd.concat(rows)
D['state'] = D['uid'].map(ust)
ly = D['uid'].map(L)
D['exit'] = ly.notna() & ((D['year'] == ly) | (D['year'] == ly + 1))
DIV = {'New England': 'CT ME MA NH RI VT', 'Mid-Atlantic': 'NJ NY PA',
       'East North Central': 'IL IN MI OH WI',
       'West North Central': 'IA KS MN MO NE ND SD',
       'South Atlantic': 'DE DC FL GA MD NC SC VA WV',
       'East South Central': 'AL KY MS TN',
       'West South Central': 'AR LA OK TX',
       'Mountain': 'AZ CO ID MT NV NM UT WY', 'Pacific': 'CA OR WA'}
s2d = {s: d for d, ss in DIV.items() for s in ss.split()}
D['division'] = D['state'].map(s2d)

def panel(region, v):
    d = D[(D['var'] == v) & D[region].notna()]
    g = d.groupby([region, 'year', 'exit'])['d'].sum().unstack(fill_value=0)
    g = g.reindex(columns=[False, True], fill_value=0)
    g.columns = ['other', 'exit']
    g = g.reset_index()
    full = pd.MultiIndex.from_product([g[region].unique(),
                                       range(min(yrs) + 1, max(yrs) + 1)])
    g = g.set_index([region, 'year']).reindex(full, fill_value=0) \
        .rename_axis([region, 'year']).reset_index()
    g = g.sort_values([region, 'year'])
    g['exit_l1'] = g.groupby(region)['exit'].shift(1)
    g['exit_f1'] = g.groupby(region)['exit'].shift(-1)
    return g.dropna()

def demean(df, cols, fes, it=200):
    X = df[cols].astype(float).copy()
    for _ in range(it):
        old = X.values.copy()
        for f in fes:
            X = X - X.groupby(df[f]).transform('mean')
        if np.abs(X.values - old).max() < 1e-9 * (1 + np.abs(old).max()):
            break
    return X

def fit(g, region, fes, xs=('exit', 'exit_l1', 'exit_f1')):
    xs = list(xs)
    Z = demean(g, ['other'] + xs, fes)
    y, X = Z['other'].values, Z[xs].values
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    cl = pd.factorize(g[region])[0]; G = cl.max() + 1
    meat = np.zeros((len(xs), len(xs)))
    for c in range(G):
        s = X[cl == c].T @ e[cl == c]
        meat += np.outer(s, s)
    V = XtX @ meat @ XtX * G / (G - 1)
    return b, V, X, y, e, cl, G, XtX

WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1,
                 np.sqrt(1.5)])
def wild_offset_p(g, region, fes, nb=9999):
    """Wild cluster bootstrap p for offset = 0 (b0 + b1 = 0), Webb weights,
    null imposed by refitting with the restriction."""
    b, V, X, y, e, cl, G, XtX = fit(g, region, fes)
    r = np.array([1.0, 1.0, 0.0])
    t0 = (r @ b) / np.sqrt(r @ V @ r)
    # restricted fit: b1 = -b0
    Xr = np.column_stack([X[:, 0] - X[:, 1], X[:, 2]])
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fr = Xr @ br; er = y - fr
    ts = np.empty(nb)
    for i in range(nb):
        w = rng.choice(WEBB, G)[cl]
        ys = fr + w * er
        bs = XtX @ X.T @ ys; es = ys - X @ bs
        meat = np.zeros((3, 3))
        for c in range(G):
            s = X[cl == c].T @ es[cl == c]
            meat += np.outer(s, s)
        Vs = XtX @ meat @ XtX * G / (G - 1)
        ts[i] = (r @ bs) / np.sqrt(r @ Vs @ r)
    return (np.abs(ts) >= abs(t0)).mean()

res = []
print('Offset share = -(b0 + b1): share of the exit step replaced by other '
      'units in the same region (same year and the next).')
for region in ['state', 'division']:
    for fes in [['year'], ['year', region]]:
        lab = f'{region}, {"year FE" if len(fes) == 1 else "year + region FE"}'
        print(f'\n{lab}:', flush=True)
        for v in VARS:
            g = panel(region, v)
            b, V, *_ = fit(g, region, fes)
            off = -(b[0] + b[1]); se = np.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1])
            p_w = wild_offset_p(g, region, fes, 1999) if region == 'division' \
                else np.nan
            res.append(dict(region=region, fe='+'.join(fes), var=v,
                            offset=off, se=se, b0=b[0], b1=b[1],
                            lead=b[2], lead_se=np.sqrt(V[2, 2]),
                            p_wild=p_w, n=len(g)))
            print(f'  {v}: offset {off:.2f} (SE {se:.2f}'
                  + (f', wild p {p_w:.3f}' if region == 'division' else '')
                  + f'); same year {-b[0]:.2f}, next year {-b[1]:.2f}; '
                  f'lead (placebo) {-b[2]:.2f} (SE {np.sqrt(V[2, 2]):.2f})')
R = pd.DataFrame(res)
R.to_csv(f'{OUT}/replacement_offsets.csv', index=False)

# what it does to the paper's counterfactual comparison
cf = f'{DIR}/counterfactual_paths_v2.csv'
for region in ['state', 'division']:
    th = R[(R['region'] == region) & (R['var'] == 'deaths')
           & (R['fe'] == 'year')]['offset'].iloc[0]
    th = min(max(th, 0), 0.99)
    msg = (f'\nDeaths-weighted offset ({region}, year FE): {th:.2f}. Net '
           f'benefit of retirement = {1 - th:.2f} x gross.')
    if os.path.exists(cf):
        C = pd.read_csv(cf)
        f, c, t = C.deaths_final.sum(), C.deaths_cf.sum(), C.deaths_t10.sum()
        msg += (f' Against the net counterfactual, the last-full-year rule '
                f'overstates {f / (c * (1 - th)):.2f}x and the ten-years-'
                f'before rule {t / (c * (1 - th)):.2f}x (gross: {f / c:.2f}x, '
                f'{t / c:.2f}x).')
    print(msg)
print('\nCaveat: replacement outside the region is not captured, so these '
      'offsets are lower bounds for national replacement.')
