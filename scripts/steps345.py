# Steps 2b, 4, 5 (emissions, model output, record presence and background
# sulfate levels only; nothing near an exit plant is read):
# (A) sulfate power in relative terms, using each event's state background
#     sulfate level (monitors > 50 km from any coal plant) as the level;
# (B) SO2 manipulation check: coverage of SO2 monitors near the exits;
# (C) counterfactual emissions path from matched non-retiring coal units,
#     and the valuation against it.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, zipfile, requests
import numpy as np, pandas as pd

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
from math import erf
Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))

# ---------- (A) sulfate power, relative terms ----------
C = pd.read_csv(f'{DIR}/so4_event_coverage.csv')
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
C = C.merge(E[['fid', 'event_year', 'state', 'lat', 'lon']],
            on=['fid', 'event_year'])
prim = C[(C['n_near_bg'] > 0) & (C['so2_drop'] >= 5000)].copy()
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
q = q[(q['ndays'] >= 3) & (q['so4'] > 0)]
q['yr'] = q['ym'] // 100
st = q.groupby('site')[['lat', 'lon', 'state']].first()
Sx = st[['lat', 'lon']].values
md = np.full(len(Sx), np.inf)
for la, lo in fc[['latitude', 'longitude']].values:
    md = np.minimum(md, km(Sx[:, 0], Sx[:, 1], la, lo))
bg_sites = set(st.index[md > 50])
bgq = q[q['site'].isin(bg_sites)]
lvl = bgq.groupby(['state', 'yr'])['so4'].mean()
# event's state code in AQS numbering: take it from the nearest sulfate site
def aqs_state(lat, lon):
    d = km(Sx[:, 0], Sx[:, 1], lat, lon)
    return st['state'].iloc[int(np.argmin(d))]
prim['aqs_state'] = [aqs_state(a, o) for a, o in zip(prim['lat'],
                                                     prim['lon'])]
prim['bg_level'] = [lvl.get((s, y - 1), np.nan) for s, y in
                    zip(prim['aqs_state'], prim['event_year'])]
prim['pred_rel'] = prim['pred_so4'] / prim['bg_level']
pl = pd.read_csv(f'{DIR}/so4_placebo_draws.csv')
sd = pl['d'].std()
se = sd / np.sqrt(len(prim))
print(f'(A) Sulfate, {len(prim)} events. State background sulfate in the '
      f'year before (µg/m³): median {prim.bg_level.median():.2f}, range '
      f'{prim.bg_level.min():.2f}-{prim.bg_level.max():.2f}')
for up in [1.0, 1.25]:
    m = (prim['pred_so4'] / (prim['bg_level'] * up)).mean()
    print(f'  Near-monitor level = {up:.2f} x background: mean predicted '
          f'relative fall {100 * m:.1f}%; SE {100 * se:.1f}%; '
          f'{m / se:.1f} SE; power (two-sided 5%) about '
          f'{100 * (Phi(m / se - 1.96) + Phi(-m / se - 1.96)):.0f}%')
prim.to_csv(f'{DIR}/so4_primary_events.csv', index=False)

# ---------- (B) SO2 monitor coverage ----------
SO = f'{DIR}/so2'
os.makedirs(SO, exist_ok=True)
for y in range(2004, 2025):
    p = f'{SO}/so2_monthly_{y}.csv'
    if os.path.exists(p):
        continue
    r = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/daily_42401_{y}.zip',
                     timeout=1200)
    if not r.ok:
        print(y, 'SO2 not available', r.status_code); continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False,
                    usecols=['State Code', 'County Code', 'Site Num',
                             'Latitude', 'Longitude', 'Date Local',
                             'Arithmetic Mean'])
    d['site'] = (d['State Code'].astype(str) + '-' + d['County Code']
                 .astype(str) + '-' + d['Site Num'].astype(str))
    d['ym'] = d['Date Local'].str[:7].str.replace('-', '').astype(int)
    m = (d.groupby(['site', 'ym'])
          .agg(lat=('Latitude', 'first'), lon=('Longitude', 'first'),
               state=('State Code', 'first'), so2=('Arithmetic Mean', 'mean'),
               ndays=('Arithmetic Mean', 'size')).reset_index())
    m.to_csv(p, index=False)
    print(y, 'SO2 sites', m['site'].nunique(), flush=True)
s2 = pd.concat([pd.read_csv(f'{SO}/so2_monthly_{y}.csv',
                            usecols=['site', 'ym', 'lat', 'lon', 'ndays'])
                for y in range(2004, 2025)])
s2 = s2[s2['ndays'] >= 8]
s2['yr'] = s2['ym'] // 100
ss = s2.groupby('site')[['lat', 'lon']].first()
ny = s2.groupby(['site', 'yr']).size()
yrs = ny[ny >= 8].reset_index().groupby('site')['yr'].apply(set)
P = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
pp = P[(P['n_near_bg'] > 0) & (P['so2_drop'] >= 5000)].merge(
    E[['fid', 'event_year', 'lat', 'lon']], on=['fid', 'event_year'])
cov = []
for r in pp.itertuples():
    d = km(ss['lat'].values, ss['lon'].values, r.lat, r.lon)
    n = sum(1 for s in ss.index[d <= 25]
            if {r.event_year - 1, r.event_year + 1} <= yrs.get(s, set()))
    cov.append(n)
pp['so2_monitors_25km'] = cov
print(f'\n(B) Primary PM2.5 events with an SO2 monitor within 25 km '
      f'(8+ months in both years): {(pp.so2_monitors_25km > 0).sum()} of '
      f'{len(pp)}')
pp[['fid', 'name', 'event_year', 'so2_monitors_25km']].to_csv(
    f'{DIR}/so2_check_coverage.csv', index=False)

# ---------- (C) counterfactual emissions path ----------
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
state = a.drop_duplicates('uid').set_index('uid')['statecode']
lastcoal = a[a['coal']].groupby('uid')['year'].max()
U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
ST2T = 0.90718474
H = A['heatinput']; S2 = A['so2mass']; NX = A['noxmass']
rows = []
for u in U.itertuples():
    T = u.last_coal_year
    if T - 5 < 1997 or T + 5 > 2025:
        continue
    # controls: coal units burning coal through T+5, positive heat t-5..t-1
    pool = [v for v in lastcoal.index[lastcoal >= T + 5]
            if v != u.uid and (H.loc[v, T - 5:T - 1] > 0).all()]
    if len(pool) < 10:
        continue
    same = [v for v in pool if state.get(v) == state.get(u.uid)]
    pool = same if len(same) >= 5 else pool
    h1, h5 = H.at[u.uid, T - 1], H.at[u.uid, T - 5]
    if h1 <= 0 or h5 <= 0:
        continue
    x0 = np.array([np.log(h1 / h5), np.log(h1),
                   np.log((S2.at[u.uid, T - 1] + 1) / h1)])
    X = np.column_stack([np.log(H.loc[pool, T - 1] / H.loc[pool, T - 5]),
                         np.log(H.loc[pool, T - 1]),
                         np.log((S2.loc[pool, T - 1] + 1)
                                / H.loc[pool, T - 1])])
    sdv = X.std(0) + 1e-9
    dist = np.sqrt((((X - x0) / sdv) ** 2).sum(1))
    nn = [pool[i] for i in np.argsort(dist)[:5]]
    yrs5 = list(range(T + 1, T + 6))
    r_s = (S2.loc[nn, yrs5].mean(axis=1) / S2.loc[nn, T - 1].clip(lower=1)).median()
    r_n = (NX.loc[nn, yrs5].mean(axis=1) / NX.loc[nn, T - 1].clip(lower=1)).median()
    c = int(u.cell)
    md_s, md_n = mv['MD_SO2_high'].values[c], mv['MD_NOx_high'].values[c]
    mm_s, mm_n = mv['MM_SO2_high'].values[c], mv['MM_NOx_high'].values[c]
    so2_cf, nox_cf = u.so2_final * r_s, u.nox_final * r_n
    rows.append(dict(uid=u.uid, T=T, ratio_so2=r_s, ratio_nox=r_n,
                     deaths_final=(u.so2_final * mm_s + u.nox_final * mm_n)
                     * ST2T,
                     deaths_cf=(so2_cf * mm_s + nox_cf * mm_n) * ST2T,
                     deaths_t10=(u.so2_t10 * mm_s + u.nox_t10 * mm_n) * ST2T,
                     dmg_cf=(so2_cf * md_s + nox_cf * md_n) * ST2T,
                     dmg_final=(u.so2_final * md_s + u.nox_final * md_n)
                     * ST2T))
CF = pd.DataFrame(rows)
CF.to_csv(f'{DIR}/counterfactual_paths.csv', index=False)
print(f'\n(C) Exits with a matched counterfactual (T-5 >= 1997, T+5 <= 2025):'
      f' {len(CF)}')
print('  Matched controls\' emissions over the next 5 years relative to '
      'their year before, median: SO2', round(CF.ratio_so2.median(), 2),
      ' NOx', round(CF.ratio_nox.median(), 2))
print(f'  Deaths/yr avoided by the retirements, counterfactual: '
      f'{CF.deaths_cf.sum():,.0f}; last-full-year rule: '
      f'{CF.deaths_final.sum():,.0f}; 10-years-before rule: '
      f'{CF.deaths_t10.sum():,.0f}')
print(f'  Overstatement vs counterfactual: last-full-year '
      f'{CF.deaths_final.sum() / CF.deaths_cf.sum():.2f}x, 10 years before '
      f'{CF.deaths_t10.sum() / CF.deaths_cf.sum():.2f}x')
