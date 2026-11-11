# Step 2: sulfate test feasibility.
# (a) EPA speciation monitors: monthly sulfate (PM2.5, parameter 88403),
#     2004-2024, saved to Drive;
# (b) coverage of the exit events (record presence only), and the model's
#     predicted sulfate fall at those monitors;
# (c) placebo noise at never-exit plants, using only monitors more than
#     50 km from every plant with an exit.
# No sulfate value near any exit plant is read.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, zipfile, requests
import numpy as np, pandas as pd
from pyproj import Transformer

SD = f'{DIR}/so4'
os.makedirs(SD, exist_ok=True)
NEAR, BGD = 50, 50          # km: near radius; background distance from coal
TPY = 907184.74e6 / (365.25 * 86400)

# ---------- (a) download ----------
for y in range(2004, 2025):
    p = f'{SD}/so4_monthly_{y}.csv'
    if os.path.exists(p):
        continue
    r = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/daily_SPEC_{y}.zip',
                     timeout=1200)
    if not r.ok:
        print(y, 'not available', r.status_code); continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    parts = []
    for ch in pd.read_csv(z.open(z.namelist()[0]), chunksize=1_000_000,
                          low_memory=False,
                          usecols=['State Code', 'County Code', 'Site Num',
                                   'Parameter Code', 'Latitude', 'Longitude',
                                   'Date Local', 'Arithmetic Mean']):
        parts.append(ch[ch['Parameter Code'] == 88403])
    d = pd.concat(parts)
    d['site'] = (d['State Code'].astype(str) + '-' + d['County Code']
                 .astype(str) + '-' + d['Site Num'].astype(str))
    d['ym'] = d['Date Local'].str[:7].str.replace('-', '').astype(int)
    m = (d.groupby(['site', 'ym'])
          .agg(lat=('Latitude', 'first'), lon=('Longitude', 'first'),
               state=('State Code', 'first'), so4=('Arithmetic Mean', 'mean'),
               ndays=('Arithmetic Mean', 'size')).reset_index())
    m.to_csv(p, index=False)
    print(y, 'sulfate sites', m['site'].nunique(), flush=True)

q = pd.concat([pd.read_csv(f'{SD}/so4_monthly_{y}.csv')
               for y in range(2004, 2025)
               if os.path.exists(f'{SD}/so4_monthly_{y}.csv')])
q = q[q['ndays'] >= 3]
q['yr'] = q['ym'] // 100
sites = q.groupby('site')[['lat', 'lon', 'state']].first()
ny = q.groupby(['site', 'yr']).size()
yrs = ny[ny >= 8].reset_index().groupby('site')['yr'].apply(set)
print('Sulfate sites with any qualifying year:', len(yrs))

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
Sx = sites[['lat', 'lon']].values
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
def min_dist(pts):
    m = np.full(len(Sx), np.inf)
    for la, lo in pts:
        m = np.minimum(m, km(Sx[:, 0], Sx[:, 1], la, lo))
    return m
sites['d_coal'] = min_dist(fc[['latitude', 'longitude']].values)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ex = fc[fc['facilityid'].isin(exit_fids)]
sites['d_exit'] = min_dist(ex[['latitude', 'longitude']].values)
sites['bg'] = sites['d_coal'] > BGD

G = np.load(f'{DIR}/isrm_grid.npz')
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, Ee, S, N = G['W'], G['E'], G['S'], G['N']
def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    h = np.where((W <= x) & (x < Ee) & (S <= y) & (y < N))[0]
    return int(h[0]) if len(h) else -1
sites['cell'] = [cell_of(a, o) for a, o in Sx]

# ---------- (b) coverage and predicted sulfate fall ----------
P = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
E = E.merge(P[['fid', 'event_year', 'layer']], on=['fid', 'event_year'],
            how='left')
E['layer'] = E['layer'].fillna(2).astype(int)
E['cell'] = E['fid'].map(fc.set_index('facilityid')['cell']).fillna(-1) \
             .astype(int)
R = np.load(f'{DIR}/isrm_rows_allcoal.npz')
rc = {c: i for i, c in enumerate(R['cells'])}
res = []
for r in E.itertuples():
    if r.cell not in rc:
        continue
    b, a_ = r.event_year - 1, r.event_year + 1
    d = km(Sx[:, 0], Sx[:, 1], r.lat, r.lon)
    near = [s for s in sites.index[d <= NEAR]
            if {b, a_} <= yrs.get(s, set()) and sites.at[s, 'cell'] >= 0]
    row = R[f'pSO4_L{r.layer}'][rc[r.cell]].astype('f8') \
        * max(r.so2_drop, 0) * TPY
    vals = []
    for s in near:
        bg = [x for x in sites.index[(sites['state'] == sites.at[s, 'state'])
                                     & sites['bg'] & (sites['cell'] >= 0)]
              if {b, a_} <= yrs.get(x, set())]
        if bg:
            vals.append(row[sites.at[s, 'cell']]
                        - row[sites.loc[bg, 'cell'].values].mean())
    res.append(dict(fid=r.fid, name=r.name, event_year=r.event_year,
                    so2_drop=r.so2_drop, layer=r.layer, n_near=len(near),
                    n_near_bg=len(vals),
                    pred_so4=np.mean(vals) if vals else np.nan))
C = pd.DataFrame(res)
C.to_csv(f'{DIR}/so4_event_coverage.csv', index=False)
cov = C[C['n_near_bg'] > 0]
for t in [1000, 5000, 10000]:
    sub = cov[cov['so2_drop'] >= t]
    print(f'SO2 drop >= {t:,} t: {len(sub)} events with a sulfate monitor '
          f'within {NEAR} km and a state background; mean predicted sulfate '
          f'fall {sub.pred_so4.mean():.3f} µg/m³ (median '
          f'{sub.pred_so4.median():.3f})')
prim = cov[cov['so2_drop'] >= 5000]

# ---------- (c) placebo noise at never-exit plants ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
cy = a[a['primaryfuelinfo'].fillna('').str.contains('Coal')
       & (a['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
ctrl = fc[fc['facilityid'].isin([f for f, ys in cy.items()
                                 if set(range(2003, 2020)) <= ys
                                 and f not in exit_fids])]
allowed = set(sites.index[sites['d_exit'] > NEAR])
q = q[q['so4'] > 0].copy()
q['l'] = np.log(q['so4'])
bgm = (q[q['site'].isin(set(sites.index[sites['bg']]))]
       .groupby(['state', 'ym'])['l'].mean().rename('bgl'))
q = q.join(bgm, on=['state', 'ym']).dropna(subset=['bgl'])
q['rel'] = q['l'] - q['bgl']
ann = q.groupby(['site', 'yr']).agg(rel=('rel', 'mean'), n=('rel', 'size'),
                                    lvl=('so4', 'mean'))
ann = ann[ann['n'] >= 8]
D, LV = {}, {}
for c in ctrl.itertuples():
    d = km(Sx[:, 0], Sx[:, 1], c.latitude, c.longitude)
    near = [s for s in sites.index[d <= NEAR] if s in allowed]
    for y in range(2005, 2019):
        v, l = [], []
        for s in near:
            if (s, y - 1) in ann.index and (s, y + 1) in ann.index:
                v.append(ann.at[(s, y - 1), 'rel'] - ann.at[(s, y + 1), 'rel'])
                l.append(ann.at[(s, y - 1), 'lvl'])
        if v:
            D[(c.facilityid, y)], LV[(c.facilityid, y)] = np.mean(v), np.mean(l)
plants = sorted({f for f, _ in D})
print(f'\nNever-exit plants with an allowed sulfate monitor: {len(plants)}')
if plants and len(prim):
    rng = np.random.default_rng(20261109)
    years = prim['event_year'].values
    dr, lv = [], []
    for _ in range(1000):
        for f in plants:
            y = int(rng.choice(years))
            if (f, y) in D:
                dr.append(D[(f, y)]); lv.append(LV[(f, y)])
    dr, lv = np.array(dr), np.array(lv)
    se = dr.std() / np.sqrt(len(prim))
    lvl = lv.mean()
    print(f'Placebo change (log): mean {dr.mean():.4f}, SD {dr.std():.4f};'
          f' pre-year sulfate {lvl:.2f} µg/m³')
    print(f'Primary sample: {len(prim)} events; SE {se * lvl:.3f} µg/m³; '
          f'MDE {2.8 * se * lvl:.3f} µg/m³; mean predicted sulfate fall '
          f'{prim.pred_so4.mean():.3f} µg/m³ = '
          f'{prim.pred_so4.mean() / (se * lvl):.1f} SE')
    pd.DataFrame({'d': dr, 'lvl': lv}).to_csv(
        f'{DIR}/so4_placebo_draws.csv', index=False)
