# Step: modelled PM2.5 change from each US closure (InMAP ISRM).
# Uses plant emissions (EPA operation data), monitor locations and record
# presence. Reads NO PM2.5 values, so it is safe before the pre-registration.
DIR = '/content/drive/MyDrive/coal_closures'
OUT = '/content/isrm/out'
import os, json, time, hashlib, requests
import numpy as np, pandas as pd
from pyproj import Transformer

meta = json.load(open(f'{OUT}/meta.json'))
V = meta['vars']
def small(name):
    m = V[name]
    return np.fromfile(m['file'], dtype=m['dtype']).reshape(m['shape'])

NS = V['pSO4_L2']['shape'][0]
layer = small('Layer')
assert np.all(layer[:NS] == 0) and np.sum(layer == 0) == NS, 'grid order'
N, S, E, W = (small(k)[:NS] for k in ['N', 'S', 'E', 'W'])
pop = small('TotalPop')[:NS]
base = small('Baseline TotalPM25')[:NS]
lh = small('LayerHeight')
print('SR layers (InMAP layer index):', small('layers').tolist())
for k in sorted(set(small('layers').tolist())):
    hk = lh[layer == k]
    print(f'  InMAP layer {k}: bottom height median {np.median(hk):.0f} m')
print(f'US population in grid: {pop.sum() / 1e6:.0f} million')

# InMAP's spatial reference (Lambert conformal conic, spherical earth)
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
cx, cy = (W + E) / 2, (S + N) / 2

def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    hit = np.where((W <= x) & (x < E) & (S <= y) & (y < N))[0]
    return int(hit[0]) if len(hit) else -1

def cells_within(lat, lon, km):
    x, y = tr.transform(lon, lat)
    return np.where(np.hypot(cx - x, cy - y) <= km * 1000)[0]

# projection check: Manhattan should be a dense small cell
c = cell_of(40.758, -73.985)
if c < 0:
    raise SystemExit('Manhattan not in grid. Send me this.')
print(f'Check, Manhattan cell: pop {pop[c]:,.0f}, size '
      f'{(E[c] - W[c]) / 1000:.0f} km, baseline PM2.5 {base[c]:.1f}')
# InMAP uses its finest cells (1 km) only in dense urban areas, so a wrong
# projection would put Manhattan in a large rural cell
if (E[c] - W[c]) > 2500:
    raise SystemExit('Projection check failed. Send me this.')
inv = Transformer.from_crs(LCC, 'EPSG:4326', always_xy=True)
top = np.argsort(-pop / ((E - W) * (N - S)))[:5]
lo_, la_ = inv.transform(cx[top], cy[top])
print('Densest cells (lat, lon):',
      [(round(a, 2), round(b, 2)) for a, b in zip(la_, lo_)])

# ---------- plants, dates, controls ----------
plants = pd.read_csv(f'{DIR}/plants.csv')
dates = pd.read_csv(f'{DIR}/us_closure_dates.csv')
dates = dates[dates['clean_closure']]
us = plants.merge(dates[['plant_id', 'event_ym']], on='plant_id')
us['oris'] = us['plant_id'].astype(str).str.replace('EIA', '')
us['oris'] = us['oris'].astype(float).astype(int)
ctrl = pd.read_csv(f'{DIR}/us_operating_controls.csv')
print('Closed plants:', len(us), ' controls:', len(ctrl))

# ---------- emissions change: 12 months before vs 12 months after ----------
def window(e, sign):
    p = pd.Period(str(int(e)), 'M')
    return [int((p + sign * k).strftime('%Y%m')) for k in range(1, 13)]

raw = pd.read_csv(f'{DIR}/us_epa_monthly_raw.csv', low_memory=False)
low = {c.lower(): c for c in raw.columns}
need = ['facilityid', 'year', 'month', 'so2mass', 'noxmass']
have = all(k in low for k in need)
print('EPA raw columns:', list(raw.columns)[:30])

URL = ('https://api.epa.gov/easey/streaming-services/emissions/'
       'apportioned/monthly')

def fetch(fids, years):
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    parts = []
    for fid in fids:
        for tries in range(6):
            r = requests.get(URL, params=dict(
                api_key=key, facilityId=int(fid),
                year='|'.join(str(y) for y in years),
                month='|'.join(str(m) for m in range(1, 13))), timeout=120)
            if r.status_code == 429:
                time.sleep(10 * (tries + 1)); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json())
        if len(d):
            d.columns = [c.lower() for c in d.columns]
            parts.append(d[need])
    return pd.concat(parts) if parts else pd.DataFrame(columns=need)

if have:
    em = raw.rename(columns={low[k]: k for k in need})[need]
else:
    em = fetch(us['oris'], range(2012, 2027))
em = em.dropna(subset=['facilityid'])
em['facilityid'] = em['facilityid'].astype(float).astype(int)
em['ym'] = em['year'].astype(int) * 100 + em['month'].astype(int)

# plants whose 12-month pre-closure window starts before the raw file does
first = em.groupby('facilityid')['ym'].min()
gap = [p.oris for p in us.itertuples()
       if p.oris not in first.index
       or window(p.event_ym, -1)[-1] < first[p.oris]]
if gap:
    print('Fetching earlier EPA months for facilities:', gap)
    ex = fetch(gap, range(2012, 2017))
    if len(ex):
        ex['facilityid'] = ex['facilityid'].astype(float).astype(int)
        ex['ym'] = ex['year'].astype(int) * 100 + ex['month'].astype(int)
        ex = ex[ex['ym'] < ex['facilityid'].map(first).fillna(999999)]
        em = pd.concat([em, ex])

fm = (em.groupby(['facilityid', 'ym'])[['so2mass', 'noxmass']]
        .sum(min_count=1).fillna(0))
fac = {k: v.droplevel(0) for k, v in fm.groupby(level=0)}
empty = pd.DataFrame(columns=['so2mass', 'noxmass'], dtype=float)

rows = []
for _, p in us.iterrows():
    g = fac.get(p['oris'], empty)
    pre = g.reindex(window(p['event_ym'], -1)).fillna(0).sum()
    post = g.reindex(window(p['event_ym'], 1)).fillna(0).sum()
    rows.append(dict(plant_id=p['plant_id'], so2_pre=pre['so2mass'],
                     nox_pre=pre['noxmass'], so2_post=post['so2mass'],
                     nox_post=post['noxmass']))
emis = pd.DataFrame(rows)
emis['d_so2'] = emis['so2_pre'] - emis['so2_post']
emis['d_nox'] = emis['nox_pre'] - emis['nox_post']
us = us.merge(emis, on='plant_id')
print('\nPlants with zero pre-closure SO2 and NOx (check):',
      int(((us['so2_pre'] == 0) & (us['nox_pre'] == 0)).sum()))

TPY = 907184.74e6 / (365.25 * 86400)    # µg/s per short ton per year

# ---------- monitors: locations and record presence only ----------
frames = [pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                      usecols=['site', 'ym', 'lat', 'lon', 'state', 'ndays'])
          for y in range(2016, 2026)]
a = pd.concat(frames)
a = a[a['ndays'] >= 8]
sites = a.groupby('site')[['lat', 'lon', 'state']].first()
months_by_site = a.groupby('site')['ym'].apply(set)

def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = (np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2)
         * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

coal = pd.concat([ctrl[['lat', 'lon']],
                  plants[plants['country'] == 'United States'][['lat', 'lon']]])
Sxy = sites[['lat', 'lon']].values
mind = np.full(len(Sxy), np.inf)
for la, lo in coal.values:
    mind = np.minimum(mind, km(Sxy[:, 0], Sxy[:, 1], la, lo))
sites['bg'] = mind > 50
sites['cell'] = [cell_of(la, lo) for la, lo in Sxy]
print('Monitors outside the grid:', int((sites['cell'] < 0).sum()))

# ---------- the matrix rows ----------
def mm(name):
    m = V[name]
    return np.memmap(m['file'], dtype=m['dtype'], mode='r',
                     shape=tuple(m['shape']))
SR = {(sp, L): mm(f'{sp}_L{L}') for sp in ['pSO4', 'pNO3', 'PrimaryPM25']
      for L in (1, 2)}

us['cell'] = [cell_of(r.lat, r.lon) for r in us.itertuples()]
ctrl['cell'] = [cell_of(r.lat, r.lon) for r in ctrl.itertuples()]
print('Plants outside the grid:', int((us['cell'] < 0).sum()),
      int((ctrl['cell'] < 0).sum()))

out, mon_rows = [], []
for _, p in us.iterrows():
    s = p['cell']
    rec = {}
    for L in (1, 2):
        so4 = np.asarray(SR[('pSO4', L)][s], dtype='f8')
        no3 = np.asarray(SR[('pNO3', L)][s], dtype='f8')
        ppm = np.asarray(SR[('PrimaryPM25', L)][s], dtype='f8')
        rec[L] = dict(so4=so4 * p['d_so2'] * TPY,
                      no3=no3 * p['d_nox'] * TPY,
                      ppm_per_kt=ppm * 1000 * TPY)
    pre = set(window(p['event_ym'], -1))
    post = set(window(p['event_ym'], 1))
    d = km(Sxy[:, 0], Sxy[:, 1], p['lat'], p['lon'])
    near = [st for st in sites.index[d <= 25]
            if len(months_by_site[st] & pre) >= 8
            and len(months_by_site[st] & post) >= 8]
    r25 = cells_within(p['lat'], p['lon'], 25)
    row = dict(plant_id=p['plant_id'], name=p['name'],
               event_ym=int(p['event_ym']), cell=s,
               d_so2_tpy=p['d_so2'], d_nox_tpy=p['d_nox'],
               n_near=len(near))
    for L, tag in [(2, 'high'), (1, 'low')]:
        tot = rec[L]['so4'] + rec[L]['no3']
        row[f'plantcell_ug_{tag}'] = tot[s]
        row[f'ring25_ug_{tag}'] = tot[r25].mean()
    contr = {2: [], 1: []}
    for st in near:
        stt = sites.at[st, 'state']
        bgs = sites.index[(sites['state'] == stt) & sites['bg']]
        bgs = [b for b in bgs if len(months_by_site[b] & (pre | post)) >= 8]
        if not bgs:
            continue
        mc = sites.at[st, 'cell']
        bc = sites.loc[bgs, 'cell'].values
        bc = bc[bc >= 0]
        if mc < 0 or len(bc) == 0:
            continue
        m = dict(plant_id=p['plant_id'], site=st,
                 dist_km=round(float(d[sites.index.get_loc(st)]), 1),
                 n_bg=len(bgs), inmap_base=base[mc])
        for L, tag in [(2, 'high'), (1, 'low')]:
            R = rec[L]
            for comp in ['so4', 'no3', 'ppm_per_kt']:
                m[f'{comp}_{tag}'] = R[comp][mc] - R[comp][bc].mean()
            m[f'total_{tag}'] = m[f'so4_{tag}'] + m[f'no3_{tag}']
        mon_rows.append(m)
        contr[2].append(m['total_high'])
        contr[1].append(m['total_low'])
    row['n_near_with_bg'] = len(contr[2])
    for L, tag in [(2, 'high'), (1, 'low')]:
        row[f'contrast_ug_{tag}'] = (np.mean(contr[L]) if contr[L]
                                     else np.nan)
    out.append(row)

pred = pd.DataFrame(out)
mon = pd.DataFrame(mon_rows)
if len(mon):
    k = (mon.groupby('plant_id')
            .agg(ppm_per_kt_high=('ppm_per_kt_high', 'mean'),
                 inmap_base=('inmap_base', 'mean')))
    pred = pred.merge(k, on='plant_id', how='left')
    pred['contrast_pct_of_inmap_base'] = (100 * pred['contrast_ug_high']
                                          / pred['inmap_base'])
pred.to_csv(f'{DIR}/model_predictions_us.csv', index=False)
mon.to_csv(f'{DIR}/model_predictions_us_monitors.csv', index=False)

# keep the matrix rows for every plant cell on Drive (small), so later
# steps never need the 66 GB again
cells = sorted((set(us['cell']) | set(ctrl['cell'])) - {-1})
np.savez_compressed(
    f'{DIR}/isrm_rows_plants.npz', cells=np.array(cells),
    **{f'{sp}_L{L}': np.asarray(SR[(sp, L)][cells])
       for (sp, L) in SR})
np.savez_compressed(f'{DIR}/isrm_grid.npz', N=N, S=S, E=E, W=W,
                    pop=pop, base=base)

h = hashlib.sha256(open(f'{DIR}/model_predictions_us.csv', 'rb')
                   .read()).hexdigest()
pd.set_option('display.width', 200)
sm = pred[pred['n_near_with_bg'] > 0]
print(f'\nPlants with monitor and background: {len(sm)}')
print('Modelled fall in PM2.5, near monitors minus state background, µg/m³ '
      '(SO2 + NOx only; positive = cleaner):')
print(sm[['name', 'event_ym', 'd_so2_tpy', 'd_nox_tpy', 'n_near_with_bg',
          'contrast_ug_high', 'contrast_ug_low']]
      .round(4).sort_values('contrast_ug_high').to_string(index=False))
print('\nMean contrast, high stack:', round(sm['contrast_ug_high'].mean(), 4),
      ' low stack:', round(sm['contrast_ug_low'].mean(), 4))
print('Mean as % of InMAP baseline PM2.5:',
      round(sm['contrast_pct_of_inmap_base'].mean(), 2))
print('All 68: mean 25 km ring change, high stack:',
      round(pred['ring25_ug_high'].mean(), 4))
print('\nSHA-256 of model_predictions_us.csv:', h)
