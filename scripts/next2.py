# (1) Europe: station coverage around large unit exits via OpenAQ v3
#     (EEA regulatory stations are included there); metadata only.
# (2) Hourly downwind design: power check at never-exit plants only.
#     Hourly wind (ERA5-Land) at those plants + hourly PM2.5 (EPA AQS) at
#     monitors within 25 km of them and > 25 km from every exit plant.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, time, zipfile, requests
import numpy as np, pandas as pd
from math import erf
Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
def bearing(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = np.sin(o2 - o1) * np.cos(a2)
    y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(o2 - o1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

# ---------- (1) Europe coverage ----------
from google.colab import userdata
KEY = userdata.get('OPENAQ_KEY')
f5 = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv',
                 usecols=['LCPInspireId', 'Latitude', 'Longitude'],
                 low_memory=False).drop_duplicates('LCPInspireId')
EU = pd.read_csv(f'{DIR}/eu_unit_exits.csv').merge(
    f5, left_on='lcp', right_on='LCPInspireId', how='left')
EU = EU[EU['last_coal_year'].between(2017, 2022)]
EU = (EU.sort_values('unit_mw_th', ascending=False)
        .drop_duplicates(['site', 'last_coal_year']))
EU['so2_drop'] = EU['site_so2_before'] - EU['site_so2_after']
EU['nox_drop'] = EU['site_nox_before'] - EU['site_nox_after']
big = EU[(EU['so2_drop'] >= 1000) | (EU['nox_drop'] >= 1000)]
rows = []
for r in big.itertuples():
    for tries in range(5):
        q = requests.get('https://api.openaq.org/v3/locations',
                         params={'coordinates': f'{r.Latitude},{r.Longitude}',
                                 'radius': 25000, 'limit': 1000},
                         headers={'X-API-Key': KEY}, timeout=60)
        if q.status_code == 429:
            time.sleep(10 * (tries + 1)); continue
        break
    locs = q.json().get('results', []) if q.ok else []
    t0 = f'{r.last_coal_year - 1}-01-01'; t1 = f'{r.last_coal_year + 1}-12-31'
    cnt = {'pm25': 0, 'so2': 0, 'no2': 0}
    for L in locs:
        first = ((L.get('datetimeFirst') or {}).get('utc') or '9999')[:10]
        last = ((L.get('datetimeLast') or {}).get('utc') or '0000')[:10]
        if first > t0 or last < t1:
            continue
        ps = {s['parameter']['name'] for s in L.get('sensors', [])}
        for p in cnt:
            cnt[p] += p in ps
    rows.append(dict(name=r.name[:40], country=r.country,
                     year=r.last_coal_year, so2_drop=round(r.so2_drop),
                     nox_drop=round(r.nox_drop), **cnt))
    time.sleep(1)
C = pd.DataFrame(rows)
C.to_csv(f'{DIR}/eu_station_coverage.csv', index=False)
pd.set_option('display.width', 200)
print('EUROPE: stations within 25 km covering the years before and after')
print(C.to_string(index=False))
print('Events with PM2.5:', int((C.pm25 > 0).sum()), ' SO2:',
      int((C.so2 > 0).sum()), ' NO2:', int((C.no2 > 0).sum()), 'of', len(C))

# ---------- (2) hourly downwind, never-exit plants ----------
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
       .dropna(subset=['latitude', 'longitude'])
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
cy = a[a['primaryfuelinfo'].fillna('').str.contains('Coal')
       & (a['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ctrl = fc[fc['facilityid'].isin([f for f, ys in cy.items()
                                 if set(range(2003, 2020)) <= ys
                                 and f not in exit_fids])]
ex = fc[fc['facilityid'].isin(exit_fids)]
mon = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                             usecols=['site', 'lat', 'lon'])
                 for y in range(2003, 2020)]).drop_duplicates('site')
Mx = mon[['lat', 'lon']].values
dex = np.full(len(Mx), np.inf)
for la, lo in ex[['latitude', 'longitude']].values:
    dex = np.minimum(dex, km(Mx[:, 0], Mx[:, 1], la, lo))
pairs = []
for c in ctrl.itertuples():
    d = km(Mx[:, 0], Mx[:, 1], c.latitude, c.longitude)
    for i in np.where((d <= 25) & (dex > 25))[0]:
        pairs.append(dict(facilityid=c.facilityid, site=mon['site'].iloc[i],
                          bearing=bearing(c.latitude, c.longitude,
                                          Mx[i, 0], Mx[i, 1])))
pairs = pd.DataFrame(pairs)
keep = set(pairs['site'])
cp = ctrl[ctrl['facilityid'].isin(set(pairs['facilityid']))]
print(f'\nHOURLY DOWNWIND: {len(cp)} never-exit plants, {len(keep)} monitors')

p_w = f'{DIR}/wind_hourly_controls.csv'
if not os.path.exists(p_w):
    import ee
    try:
        ee.Initialize(project='coal-closures-air')
    except Exception:
        ee.Authenticate(); ee.Initialize(project='coal-closures-air')
    col = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select(
        ['u_component_of_wind_10m', 'v_component_of_wind_10m'])
    pts = ee.Geometry.MultiPoint(cp[['longitude', 'latitude']].values
                                 .tolist())
    WD = f'{DIR}/windh'
    os.makedirs(WD, exist_ok=True)
    for y in range(2003, 2020):
        py = f'{WD}/windh_{y}.csv'
        if os.path.exists(py):
            continue
        parts = []
        for m0 in range(1, 13):
            s0 = f'{y}-{m0:02d}-01'
            s1 = f'{y + (m0 == 12)}-{m0 % 12 + 1:02d}-01'
            for tries in range(4):
                try:
                    reg = col.filterDate(s0, s1).getRegion(pts, 11132) \
                             .getInfo(); break
                except Exception as e_:
                    print('  retry', s0, str(e_)[:60]); time.sleep(15)
            parts.append(pd.DataFrame(reg[1:], columns=reg[0]))
        pd.concat(parts).to_csv(py, index=False)
        print(y, 'hourly wind done', flush=True)
    w = pd.concat([pd.read_csv(f'{WD}/windh_{y}.csv')
                   for y in range(2003, 2020)]).dropna()
    px = w[['longitude', 'latitude']].drop_duplicates().values
    m = []
    for f in cp.itertuples():
        dd = (px[:, 0] - f.longitude) ** 2 + (px[:, 1] - f.latitude) ** 2
        j = int(np.argmin(dd))
        if dd[j] <= 0.2 ** 2:
            m.append((f.facilityid, px[j, 0], px[j, 1]))
    w = pd.DataFrame(m, columns=['facilityid', 'longitude', 'latitude']) \
          .merge(w, on=['longitude', 'latitude'])
    w['hour'] = pd.to_datetime(w['time'], unit='ms').dt.strftime(
        '%Y-%m-%d %H')
    w = w.rename(columns={'u_component_of_wind_10m': 'u',
                          'v_component_of_wind_10m': 'v'})
    w[['facilityid', 'hour', 'u', 'v']].to_csv(p_w, index=False)
w = pd.read_csv(p_w)

p_h = f'{DIR}/pm25_hourly_controls.csv'
if not os.path.exists(p_h):
    parts = []
    for y in range(2003, 2020):
        r = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/'
                         f'hourly_88101_{y}.zip', timeout=1800)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        got = 0
        for ch in pd.read_csv(z.open(z.namelist()[0]), chunksize=2_000_000,
                              low_memory=False,
                              usecols=['State Code', 'County Code',
                                       'Site Num', 'Date GMT', 'Time GMT',
                                       'Sample Measurement']):
            ch['site'] = (ch['State Code'].astype(str) + '-'
                          + ch['County Code'].astype(str) + '-'
                          + ch['Site Num'].astype(str))
            ch = ch[ch['site'].isin(keep)]
            ch['hour'] = ch['Date GMT'] + ' ' + ch['Time GMT'].str[:2]
            parts.append(ch.groupby(['site', 'hour'])['Sample Measurement']
                         .mean().reset_index())
            got += len(ch)
        print(y, 'hourly PM2.5 rows kept', got, flush=True)
    pd.concat(parts).rename(columns={'Sample Measurement': 'pm'}) \
        .to_csv(p_h, index=False)
ph = pd.read_csv(p_h)
ph = ph[ph['pm'] > 0]
ph['l'] = np.log(ph['pm'])
ph['yr'] = ph['hour'].str[:4].astype(int)

w['to'] = (np.degrees(np.arctan2(w['u'], w['v'])) + 360) % 360
X = pairs.merge(ph, on='site').merge(w[['facilityid', 'hour', 'to']],
                                     on=['facilityid', 'hour'])
diff = np.abs((X['to'] - X['bearing'] + 180) % 360 - 180)
X['dir'] = np.where(diff <= 45, 'down', np.where(diff >= 135, 'up', 'side'))
g = X[X['dir'] != 'side'].groupby(['facilityid', 'site', 'yr', 'dir'])
s = g['l'].agg(['mean', 'size']).unstack('dir')
s = s[(s[('size', 'down')] >= 100) & (s[('size', 'up')] >= 100)]
s['du'] = s[('mean', 'down')] - s[('mean', 'up')]
du = s['du'].reset_index()
du.columns = ['facilityid', 'site', 'yr', 'du']
share = (X['dir'] == 'down').mean()
lvl = X['pm'].mean()
D = {}
for (f, si), gg in du.groupby(['facilityid', 'site']):
    gg = gg.set_index('yr')['du']
    for y in range(2004, 2019):
        if y - 1 in gg.index and y + 1 in gg.index:
            D.setdefault((f, y), []).append(gg[y - 1] - gg[y + 1])
D = {k: np.mean(v) for k, v in D.items()}
plants = sorted({f for f, _ in D})
P2 = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
for thr in [2500, 5000]:
    prim = P2[(P2['n_near_bg'] > 0) & (P2['so2_drop'] >= thr)]
    years = prim['event_year'].values
    rng = np.random.default_rng(20261109)
    dr = np.array([D[(f, int(y))] for _ in range(1000) for f in plants
                   for y in [rng.choice(years)] if (f, int(y)) in D])
    se = dr.std() / np.sqrt(len(prim)) * lvl
    pred = prim['pred_main'].mean() / share
    z = pred / se
    print(f'Threshold {thr}: {len(prim)} events; hourly placebo SD '
          f'{dr.std():.4f} log ({len(plants)} plants); SE {se:.3f} µg/m³; '
          f'rough downwind prediction {pred:.2f} µg/m³ = {z:.1f} SE; '
          f'power vs zero about {100 * (Phi(z - 1.96) + Phi(-z - 1.96)):.0f}%')
print(f'Share of downwind hours {share:.2f}; mean PM2.5 {lvl:.1f} µg/m³')
pd.DataFrame({'d': dr}).to_csv(f'{DIR}/downwind_hourly_placebo_draws.csv',
                               index=False)
