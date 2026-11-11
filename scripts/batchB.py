# POST-REGISTRATION, exploratory. Batch B, 10 Nov 2026.
#  (A) Stacked event study of monthly PM2.5 around US coal plant closures
#      (Cengiz, Dube, Lindner and Zipperer 2019). Each closure is its own
#      stack: treated monitors lie within r miles of the closing plant and
#      within r of no other closure in the stack's window; control monitors
#      lie in the same state, between 2r and 250 km from the plant, and
#      within r of no closure in the window. Monitor-by-stack and
#      month-by-stack fixed effects; quarters -8..+8 relative to quarter -1;
#      standard errors clustered by monitor. This removes the comparisons
#      between early- and late-treated monitors that a pooled two-way
#      fixed-effects event study makes when closures are staggered.
#      Reported: the event-time path, the step at closure (quarters 0..3
#      minus quarters -4..-1), the decline before closure (quarters -8..-5
#      minus -4..-1) and a stacked post-closure indicator; all closures and
#      large closures (plant SO2 at least 5,000 tons in the last full year).
#  (B) Map data for the design figure: exit events (with sample flags) and
#      PM2.5 monitors (near an event plant / background / other).
# Same closure dating and monitor data as btdesign.py.
import os, hashlib
import numpy as np, pandas as pd
DIR = os.environ.get('CC_DIR', '/content/drive/MyDrive/coal_closures')
OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
MILE = 1.609344
W = 24                                   # months each side (quarters -8..8)
saved = []
def save(df, name, **kw):
    p = f'{OUT}/{name}'; df.to_csv(p, **kw); saved.append(p)
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

# ---------- closures, dated as in btdesign.py ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'statecode',
                         'so2mass', 'heatinput', 'primaryfuelinfo'])
for c in ['so2mass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
cyr = a[a['coal']].groupby('facilityid')['year'].agg(['min', 'max',
                                                       'nunique'])
YMAX = int(a['year'].max())
clo = cyr[(cyr['max'] < YMAX) & (cyr['nunique'] >= 2) & (cyr['max'] >= 2000)]
coal_units = a[a['coal']].groupby('facilityid')['unitid'] \
    .apply(lambda s: set(s.astype(str)))
pstate = a.drop_duplicates('facilityid').set_index('facilityid')['statecode']
pso2 = a.groupby(['facilityid', 'year'])['so2mass'].sum()
mon = pd.read_csv(f'{DIR}/us_epa_monthly_closures.csv')
mon['heatinput'] = pd.to_numeric(mon['heatinput'], errors='coerce').fillna(0)
mon = mon[[str(u) in coal_units.get(int(f), set())
           for f, u in zip(mon['facilityid'], mon['unitid'])]]
lastm = mon[mon['heatinput'] > 0].assign(
    ym=lambda d: d['year'].astype(int) * 12 + d['month'].astype(int) - 1) \
    .groupby('facilityid')['ym'].max()
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
    .dropna(subset=['latitude', 'longitude'])
C = pd.DataFrame({'fid': clo.index})
C['close_m'] = C['fid'].map(lastm) + 1
miss = C['close_m'].isna()
C.loc[miss, 'close_m'] = (clo.loc[C.loc[miss, 'fid'], 'max'].values + 1) * 12
C = C.merge(fc[['facilityid', 'latitude', 'longitude']], left_on='fid',
            right_on='facilityid').reset_index(drop=True)
C['state'] = C['fid'].map(pstate)
C['so2_lfy'] = [pso2.get((f, int(m // 12) - 1), 0.0)
                for f, m in zip(C['fid'], C['close_m'])]
print(f'{len(C)} plant closures located (2000-{YMAX - 1}); '
      f'{(C.so2_lfy >= 5000).sum()} with at least 5,000 tons of SO2 in the '
      'last full year')

# ---------- monitor panel ----------
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)
               if os.path.exists(f'{DIR}/aqs/pm25_monthly_{y}.csv')])
q = q[(q['ndays'] >= 4) & (q['pm'] > 0)].copy()
q['m'] = (q['ym'] // 100) * 12 + q['ym'] % 100 - 1
st = q.groupby('site')[['lat', 'lon', 'state']].first()
FIPS = dict(zip([1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51,
                 53, 54, 55, 56],
                'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME '
                'MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI '
                'SC SD TN TX UT VT VA WA WV WI WY'.split()))
def to_postal(x):
    try:
        return FIPS.get(int(float(x)), str(x))
    except (TypeError, ValueError):
        return str(x).strip().upper()
st['state'] = st['state'].map(to_postal)
print('monitor states matched to plant states:',
      f"{st['state'].isin(set(C['state'])).mean():.0%} of monitors")
SITES = st.index.values
Dc = np.stack([km(st['lat'].values, st['lon'].values, la, lo)
               for la, lo in C[['latitude', 'longitude']].values], 1)
MMIN, MMAX = q['m'].min(), q['m'].max()
q = q.sort_values(['site', 'm'])

def demean(Y, groups, it=1000, tol=1e-9):
    Y = Y.copy()
    for _ in range(it):
        old = Y.copy()
        for g, n in groups:
            cnt = np.bincount(g, minlength=n)
            Y -= (np.stack([np.bincount(g, Y[:, j], minlength=n)
                            for j in range(Y.shape[1])], 1)
                  / cnt[:, None])[g]
        if np.abs(Y - old).max() < tol * (1 + np.abs(old).max()):
            break
    return Y
def ols_fe(d, y, xs, fes, cl):
    groups = [(pd.factorize(d[f])[0], d[f].nunique()) for f in fes]
    Z = demean(d[[y] + xs].values.astype(float), groups)
    yv, X = Z[:, 0], Z[:, 1:]
    XtX = np.linalg.pinv(X.T @ X); b = XtX @ X.T @ yv; e = yv - X @ b
    g = pd.factorize(d[cl])[0]; G = g.max() + 1
    S = np.stack([np.bincount(g, X[:, j] * e, minlength=G)
                  for j in range(X.shape[1])], 1)
    V = XtX @ (S.T @ S) @ XtX * G / (G - 1)
    return pd.Series(b, index=xs), pd.DataFrame(V, index=xs, columns=xs)
def lincomb(b, V, wts):
    w = pd.Series(wts).reindex(b.index).fillna(0.0)
    return float(w @ b), float(np.sqrt(w @ V @ w))

KS = [k for k in range(-8, 9) if k != -1]
DUM = [f'q{k}' for k in KS]
rows_sum, rows_path = [], []
for miles in [25, 50]:
    r = miles * MILE
    for sample in ['all closures', 'large closures (SO2 >= 5,000 t)']:
        stacks, n_tr, n_ct = [], 0, 0
        for j, c in C.iterrows():
            if sample.startswith('large') and c['so2_lfy'] < 5000:
                continue
            cm = int(c['close_m'])
            if cm - W < MMIN or cm + W + 2 > MMAX:
                continue
            inwin = (np.abs(C['close_m'].values - cm) <= W + 3)
            other = inwin.copy(); other[j] = False
            near_other = (Dc[:, other] <= r).any(1)
            near_any_win = (Dc[:, inwin] <= r).any(1)
            tr = (Dc[:, j] <= r) & ~near_other
            ct = (st['state'].values == c['state']) & (Dc[:, j] > 2 * r) \
                & (Dc[:, j] <= 250) & ~near_any_win
            if tr.sum() == 0 or ct.sum() < 3:
                continue
            sel = set(SITES[tr]) | set(SITES[ct])
            d = q[q['site'].isin(sel) & (q['m'] >= cm - W)
                  & (q['m'] <= cm + W + 2)][['site', 'm', 'pm']].copy()
            d['treat'] = d['site'].isin(set(SITES[tr])).astype(float)
            if d.loc[d.treat == 1, 'm'].lt(cm).sum() == 0 or \
                    d.loc[d.treat == 1, 'm'].ge(cm).sum() == 0:
                continue
            d['k'] = np.floor((d['m'] - cm) / 3).astype(int)
            d['stack'] = j
            stacks.append(d); n_tr += tr.sum(); n_ct += ct.sum()
        if len(stacks) < 5:
            print(f'\n{miles} miles, {sample}: only {len(stacks)} usable closures; skipped')
            continue
        S = pd.concat(stacks)
        S['site_stack'] = S['site'].astype(str) + '_' + S['stack'].astype(str)
        S['m_stack'] = S['m'].astype(str) + '_' + S['stack'].astype(str)
        for k, nm in zip(KS, DUM):
            S[nm] = ((S['k'] == k) & (S['treat'] == 1)).astype(float)
        S['post'] = ((S['k'] >= 0) & (S['treat'] == 1)).astype(float)
        b, V = ols_fe(S, 'pm', DUM, ['site_stack', 'm_stack'], 'site')
        bp, Vp = ols_fe(S, 'pm', ['post'], ['site_stack', 'm_stack'], 'site')
        step = lincomb(b, V, {**{f'q{k}': 0.25 for k in range(0, 4)},
                              **{f'q{k}': -0.25 for k in range(-4, -1)}})
        before = lincomb(b, V, {**{f'q{k}': 0.25 for k in range(-8, -4)},
                                **{f'q{k}': -0.25 for k in range(-4, -1)}})
        late = lincomb(b, V, {**{f'q{k}': 0.25 for k in range(5, 9)},
                              **{f'q{k}': -0.25 for k in range(-4, -1)}})
        lab = f'{miles} miles, {sample}'
        print(f'\n{lab}: {len(stacks)} closures, {n_tr} treated and {n_ct} '
              f'control monitor-stacks, {len(S):,} monitor-months; mean '
              f'PM2.5 {S.pm.mean():.2f}', flush=True)
        print(f'  step at closure (quarters 0..3 minus -4..-1): '
              f'{step[0]:+.3f} ({step[1]:.3f})')
        print(f'  two years before (quarters -8..-5 minus -4..-1): '
              f'{before[0]:+.3f} ({before[1]:.3f})')
        print(f'  two years after (quarters 5..8 minus -4..-1): '
              f'{late[0]:+.3f} ({late[1]:.3f})')
        print(f'  stacked post-closure indicator (vs quarters -8..-1): '
              f'{bp["post"]:+.3f} ({np.sqrt(Vp.iloc[0, 0]):.3f})')
        print('  path (ref quarter -1): ' + ', '.join(
            f'{k}: {b[f"q{k}"]:+.2f}' for k in KS))
        rows_sum += [dict(buffer=miles, sample=sample, closures=len(stacks),
                          stat=s, est=e[0], se=e[1])
                     for s, e in [('step 0..3 vs -4..-1', step),
                                  ('-8..-5 vs -4..-1', before),
                                  ('5..8 vs -4..-1', late),
                                  ('post indicator', (bp['post'],
                                                      np.sqrt(Vp.iloc[0, 0])))]]
        rows_path += [dict(buffer=miles, sample=sample, k=k,
                           coef=b[f'q{k}'], se=np.sqrt(V.loc[f'q{k}', f'q{k}']))
                      for k in KS] + [dict(buffer=miles, sample=sample, k=-1,
                                           coef=0.0, se=0.0)]
save(pd.DataFrame(rows_sum), 'stacked_summary.csv', index=False)
save(pd.DataFrame(rows_path).sort_values(['buffer', 'sample', 'k']),
     'stacked_eventtime.csv', index=False)

# ---------- (B) map data ----------
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
print('\n(B) exit-event columns:', list(E.columns)[:12])
pts = []
F = pd.read_csv(f'{OUT}/events_pm25_35km.csv')
key = [c for c in ['fid', 'event_year'] if c in F.columns and c in E.columns]
inpm = set(map(tuple, F[key].values)) if len(key) == 2 else set()
S4 = pd.read_csv(f'{OUT}/events_so4_100km.csv')
inso = set(map(tuple, S4[key].values)) if len(key) == 2 and \
    all(c in S4.columns for c in key) else set()
for r in E.itertuples():
    k_ = (r.fid, r.event_year)
    pts.append(dict(kind='exit event', lat=r.lat, lon=r.lon,
                    year=r.event_year,
                    so2_drop=getattr(r, 'so2_drop', np.nan),
                    pm25_sample=k_ in inpm, so4_sample=k_ in inso))
ev51 = E[[(f, y) in inpm for f, y in zip(E['fid'], E['event_year'])]]
Dn = np.stack([km(st['lat'].values, st['lon'].values, la, lo)
               for la, lo in ev51[['lat', 'lon']].values], 1) \
    if len(ev51) else np.full((len(st), 1), np.inf)
Df = np.stack([km(st['lat'].values, st['lon'].values, la, lo)
               for la, lo in fc[['latitude', 'longitude']].values], 1)
role = np.where((Dn <= 35).any(1), 'near (within 35 km of a PM2.5 event)',
                np.where(Df.min(1) > 50, 'background (over 50 km from coal)',
                         'other'))
for (s, rr), ro in zip(st.iterrows(), role):
    pts.append(dict(kind='PM2.5 monitor', lat=rr['lat'], lon=rr['lon'],
                    role=ro))
P = pd.DataFrame(pts)
save(P, 'map_points.csv', index=False)
print(f'  map points: {len(E)} exit events ({len(inpm)} in the PM2.5 '
      f'sample, {len(inso)} in the sulfate sample); {len(st)} PM2.5 '
      'monitors:', pd.Series(role).value_counts().to_dict())

print('\nSHA-256 of the new outputs:')
for p in saved:
    print(f'  {os.path.basename(p)}  '
          f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
