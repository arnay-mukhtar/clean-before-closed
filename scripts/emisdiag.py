# Step: emissions trajectories of closed and control plants, and how much
# of the modelled PM2.5 gain arrives before the closure date.
# Uses EPA operation data and the saved InMAP rows only. No outcomes.
DIR = '/content/drive/MyDrive/coal_closures'
import os, time, requests
import numpy as np, pandas as pd
from pyproj import Transformer

need = ['facilityid', 'year', 'month', 'so2mass', 'noxmass', 'heatinput']
URL = ('https://api.epa.gov/easey/streaming-services/emissions/'
       'apportioned/monthly')

def fetch(fids, years):
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    parts = []
    for i, fid in enumerate(fids):
        for tries in range(6):
            r = requests.get(URL, params=dict(
                api_key=key, facilityId=int(fid),
                year='|'.join(str(y) for y in years),
                month='|'.join(str(m) for m in range(1, 13))), timeout=180)
            if r.status_code == 429:
                time.sleep(10 * (tries + 1)); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json())
        if len(d):
            d.columns = [c.lower() for c in d.columns]
            parts.append(d[need])
        if (i + 1) % 25 == 0:
            print(f'  {i + 1} of {len(fids)} facilities', flush=True)
    return pd.concat(parts) if parts else pd.DataFrame(columns=need)

def ids(s):
    return s.astype(str).str.replace('EIA', '').astype(float).astype(int)

plants = pd.read_csv(f'{DIR}/plants.csv')
dates = pd.read_csv(f'{DIR}/us_closure_dates.csv')
us = plants.merge(dates[dates['clean_closure']][['plant_id', 'event_ym']],
                  on='plant_id')
us['oris'] = ids(us['plant_id'])
ctrl = pd.read_csv(f'{DIR}/us_operating_controls.csv')
ctrl['oris'] = ids(ctrl['plant_id'])

# closed plants: raw file plus earlier years
raw = pd.read_csv(f'{DIR}/us_epa_monthly_raw.csv', low_memory=False)
raw.columns = [c.lower() for c in raw.columns]
em = raw[need]
p_early = f'{DIR}/us_epa_monthly_early.csv'
if os.path.exists(p_early):
    early = pd.read_csv(p_early)
else:
    print('Fetching 2012-2015 for closed plants')
    early = fetch(us['oris'], range(2012, 2016))
    early.to_csv(p_early, index=False)
em = pd.concat([em[em['year'] >= 2016], early[early['year'] < 2016]])

p_ctrl = f'{DIR}/us_epa_controls_monthly.csv'
if os.path.exists(p_ctrl):
    ec = pd.read_csv(p_ctrl)
else:
    print('Fetching 2016-2025 for 149 control plants (about 10-20 min)')
    ec = fetch(ctrl['oris'], range(2016, 2026))
    ec.to_csv(p_ctrl, index=False)

def annual(e):
    e = e.dropna(subset=['facilityid']).copy()
    e['facilityid'] = e['facilityid'].astype(float).astype(int)
    for c in ['so2mass', 'noxmass', 'heatinput']:
        e[c] = pd.to_numeric(e[c], errors='coerce').fillna(0)
    e['ym'] = e['year'].astype(int) * 100 + e['month'].astype(int)
    return e.groupby(['facilityid', 'ym'])[
        ['so2mass', 'noxmass', 'heatinput']].sum()

mc, mk = annual(em), annual(ec)

def win(e, sign):
    p = pd.Period(str(int(e)), 'M')
    return [int((p + sign * k).strftime('%Y%m')) for k in range(1, 13)]

# ---------- InMAP: 25 km ring PM2.5 per kt, high and low stack ----------
G = np.load(f'{DIR}/isrm_grid.npz')
R = np.load(f'{DIR}/isrm_rows_plants.npz')
cells = list(R['cells'])
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, E, S, N = G['W'], G['E'], G['S'], G['N']
cx, cy = (W + E) / 2, (S + N) / 2
TPY = 907184.74e6 / (365.25 * 86400)

def per_kt(lat, lon, L):
    x, y = tr.transform(lon, lat)
    c = int(np.where((W <= x) & (x < E) & (S <= y) & (y < N))[0][0])
    j = cells.index(c)
    ring = np.hypot(cx - x, cy - y) <= 25000
    ring[c] = True
    so2 = R[f'pSO4_L{L}'][j][ring].mean() * 1000 * TPY
    nox = R[f'pNO3_L{L}'][j][ring].mean() * 1000 * TPY
    return so2, nox

rows = []
for _, p in us.iterrows():
    try:
        g = mc.loc[p['oris']]
    except KeyError:
        continue
    g = g.copy()
    g['yr'] = g.index // 100
    ann = g.groupby('yr')[['so2mass', 'noxmass']].sum()
    ann = ann[(ann.index >= 2012) & (ann.index <= 2025)]
    pre = g.reindex(win(p['event_ym'], -1)).fillna(0).sum()
    s_hi, n_hi = per_kt(p['lat'], p['lon'], 2)
    s_lo, n_lo = per_kt(p['lat'], p['lon'], 1)
    pk = ann['so2mass'].idxmax()
    rows.append(dict(
        name=p['name'], event_ym=int(p['event_ym']), peak_year=int(pk),
        so2_peak=ann['so2mass'].max(), nox_peak=ann['noxmass'].max(),
        so2_pre12=pre['so2mass'], nox_pre12=pre['noxmass'],
        pm_ring_peak_hi=(ann.loc[pk, 'so2mass'] * s_hi
                         + ann.loc[pk, 'noxmass'] * n_hi) / 1000,
        pm_ring_pre12_hi=(pre['so2mass'] * s_hi
                          + pre['noxmass'] * n_hi) / 1000,
        pm_ring_pre12_lo=(pre['so2mass'] * s_lo
                          + pre['noxmass'] * n_lo) / 1000))
d = pd.DataFrame(rows)
d['share_left_at_closure'] = d['so2_pre12'] / d['so2_peak'].replace(0, np.nan)
d.to_csv(f'{DIR}/closed_emission_trajectories.csv', index=False)

ck = []
for _, p in ctrl.iterrows():
    try:
        g = mk.loc[p['oris']].copy()
    except KeyError:
        continue
    g['yr'] = g.index // 100
    ann = g.groupby('yr')[['so2mass', 'noxmass']].sum()
    ann = ann[(ann.index >= 2016) & (ann.index <= 2025)]
    if len(ann) < 5:
        continue
    yoy = ann['so2mass'].diff()
    ck.append(dict(name=p['name'], so2_2016=ann['so2mass'].iloc[0],
                   so2_2025=ann['so2mass'].iloc[-1],
                   so2_mean=ann['so2mass'].mean(),
                   nox_mean=ann['noxmass'].mean(),
                   biggest_yoy_fall=-yoy.min()))
c = pd.DataFrame(ck)
c.to_csv(f'{DIR}/control_emission_summary.csv', index=False)

pd.set_option('display.width', 200)
q = lambda s: s.quantile([.25, .5, .75]).round(0).tolist()
print('\nCLOSED PLANTS (n=%d), short tons per year' % len(d))
print('  SO2 at peak year, quartiles:', q(d['so2_peak']))
print('  SO2 in last 12 months before closure:', q(d['so2_pre12']))
print('  NOx at peak year:', q(d['nox_peak']))
print('  NOx in last 12 months:', q(d['nox_pre12']))
print('  Median share of peak SO2 still emitted at closure:',
      round(d['share_left_at_closure'].median(), 3))
print('  Modelled 25 km PM2.5 from the plant, µg/m³ (high stack):')
print('    at peak year, quartiles:',
      d['pm_ring_peak_hi'].quantile([.25, .5, .75]).round(3).tolist())
print('    in last 12 months:',
      d['pm_ring_pre12_hi'].quantile([.25, .5, .75]).round(3).tolist())
print('    last 12 months, low stack:',
      d['pm_ring_pre12_lo'].quantile([.25, .5, .75]).round(3).tolist())
print('  Peak years:', d['peak_year'].value_counts().sort_index().to_dict())
print('\nCONTROL PLANTS (n=%d), short tons per year' % len(c))
print('  SO2 mean, quartiles:', q(c['so2_mean']))
print('  NOx mean, quartiles:', q(c['nox_mean']))
print('  SO2 2016 vs 2025 medians:', round(c['so2_2016'].median()),
      round(c['so2_2025'].median()))
print('  Biggest single-year SO2 fall, quartiles:', q(c['biggest_yoy_fall']))
print('\nTop 10 closed plants by SO2 in last 12 months:')
print(d.sort_values('so2_pre12', ascending=False).head(10)[
    ['name', 'event_ym', 'peak_year', 'so2_peak', 'so2_pre12',
     'pm_ring_pre12_hi']].round(3).to_string(index=False))
