# Step 3: downwind design, power check on never-exit plants.
# (a) daily 10 m wind at every US coal plant, 2003-2025 (ERA5-Land, Earth
#     Engine), saved for later use at the exit events too;
# (b) daily PM2.5 2003-2019 kept ONLY for monitors within 25 km of a
#     never-exit plant and more than 25 km from every plant with an exit;
# (c) placebo statistic: (downwind - upwind) in the year after minus the
#     year before, log PM2.5, at never-exit plants.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, time, zipfile, requests
import numpy as np, pandas as pd

fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
       .dropna(subset=['latitude', 'longitude'])

# ---------- (a) wind ----------
p_w = f'{DIR}/wind_daily_plants.csv'
if not os.path.exists(p_w):
    import ee
    try:
        ee.Initialize(project='coal-closures-air')
    except Exception:
        ee.Authenticate(); ee.Initialize(project='coal-closures-air')
    col = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(
        ['u_component_of_wind_10m', 'v_component_of_wind_10m'])
    pts = ee.Geometry.MultiPoint(fc[['longitude', 'latitude']].values
                                 .tolist())
    WD = f'{DIR}/wind'
    os.makedirs(WD, exist_ok=True)
    out = []
    for y in range(2003, 2026):
        py = f'{WD}/wind_{y}.csv'
        if not os.path.exists(py):
            parts = []
            for m0 in range(1, 13):          # one month per request
                s0 = f'{y}-{m0:02d}-01'
                s1 = f'{y + (m0 == 12)}-{m0 % 12 + 1:02d}-01'
                for tries in range(4):
                    try:
                        reg = col.filterDate(s0, s1).getRegion(pts, 11132) \
                                 .getInfo()
                        break
                    except Exception as ex:
                        print('  retry', s0, str(ex)[:60]); time.sleep(10)
                parts.append(pd.DataFrame(reg[1:], columns=reg[0]))
            pd.concat(parts).to_csv(py, index=False)
        d = pd.read_csv(py)
        out.append(d)
        print(y, 'wind rows', len(d), flush=True)
    w = pd.concat(out).dropna()
    # map pixel centres back to plants (nearest)
    px = w[['longitude', 'latitude']].drop_duplicates().values
    rows = []
    for f in fc.itertuples():
        dd = (px[:, 0] - f.longitude) ** 2 + (px[:, 1] - f.latitude) ** 2
        j = int(np.argmin(dd))
        if dd[j] > 0.2 ** 2:
            continue
        rows.append((f.facilityid, px[j, 0], px[j, 1]))
    m = pd.DataFrame(rows, columns=['facilityid', 'longitude', 'latitude'])
    w = m.merge(w, on=['longitude', 'latitude'])
    w['date'] = pd.to_datetime(w['time'], unit='ms').dt.strftime('%Y-%m-%d')
    w = w.rename(columns={'u_component_of_wind_10m': 'u',
                          'v_component_of_wind_10m': 'v'})
    w[['facilityid', 'date', 'u', 'v']].to_csv(p_w, index=False)
w = pd.read_csv(p_w)
print('Wind: plants', w['facilityid'].nunique(), ' days', w['date'].nunique())

# ---------- control plants and allowed monitors ----------
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
if pairs.empty:
    raise SystemExit('No allowed monitors near never-exit plants. Send me this.')
keep = set(pairs['site'])
print('Control plants with allowed monitors:', pairs['facilityid'].nunique(),
      ' monitors:', len(keep))

# ---------- (b) daily PM2.5 for those monitors only ----------
p_d = f'{DIR}/pm25_daily_controls.csv'
if not os.path.exists(p_d):
    parts = []
    for y in range(2003, 2020):
        r = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/'
                         f'daily_88101_{y}.zip', timeout=900)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False,
                        usecols=['State Code', 'County Code', 'Site Num',
                                 'Date Local', 'Arithmetic Mean'])
        d['site'] = (d['State Code'].astype(str) + '-' + d['County Code']
                     .astype(str) + '-' + d['Site Num'].astype(str))
        d = d[d['site'].isin(keep)]
        d = d.groupby(['site', 'Date Local'])['Arithmetic Mean'].mean() \
             .reset_index()
        parts.append(d)
        print(y, 'daily rows kept', len(d), flush=True)
    pd.concat(parts).rename(columns={'Date Local': 'date',
                                     'Arithmetic Mean': 'pm'}) \
        .to_csv(p_d, index=False)
pmd = pd.read_csv(p_d)
pmd = pmd[pmd['pm'] > 0]
pmd['l'] = np.log(pmd['pm'])
pmd['yr'] = pmd['date'].str[:4].astype(int)

# ---------- (c) placebo downwind statistic ----------
w['to'] = (np.degrees(np.arctan2(w['u'], w['v'])) + 360) % 360  # blows toward
X = pairs.merge(pmd, on='site').merge(w[['facilityid', 'date', 'to']],
                                      on=['facilityid', 'date'])
diff = np.abs((X['to'] - X['bearing'] + 180) % 360 - 180)
X['dir'] = np.where(diff <= 45, 'down', np.where(diff >= 135, 'up', 'side'))
g = X[X['dir'] != 'side'].groupby(['facilityid', 'site', 'yr', 'dir'])
s = g['l'].agg(['mean', 'size']).unstack('dir')
ok = (s[('size', 'down')] >= 10) & (s[('size', 'up')] >= 10)
s = s[ok]
s['du'] = s[('mean', 'down')] - s[('mean', 'up')]
du = s['du'].reset_index()
du.columns = ['facilityid', 'site', 'yr', 'du']
share_down = (X['dir'] == 'down').mean()
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
prim = P2[(P2['n_near_bg'] > 0) & (P2['so2_drop'] >= 5000)]
years = prim['event_year'].values
rng = np.random.default_rng(20261109)
dr = np.array([D[(f, int(y))] for _ in range(1000) for f in plants
               for y in [rng.choice(years)] if (f, int(y)) in D])
se_log = dr.std() / np.sqrt(len(prim))
pred_dw = prim['pred_main'].mean() / share_down
print(f'\nShare of days downwind: {share_down:.2f}; mean PM2.5 at these '
      f'monitors {lvl:.1f} µg/m³')
print(f'Placebo (down-up change, log): mean {dr.mean():.4f}, SD '
      f'{dr.std():.4f}, {len(dr):,} plant-draws from {len(plants)} plants')
print(f'For {len(prim)} events: SE {se_log * lvl:.3f} µg/m³, MDE '
      f'{2.8 * se_log * lvl:.2f} µg/m³')
print(f'Rough downwind-day prediction (annual main prediction / share of '
      f'downwind days): {pred_dw:.2f} µg/m³ = '
      f'{pred_dw / (se_log * lvl):.1f} SE')
print(f'Annual design for comparison: main prediction 0.158 µg/m³ = 0.6 SE')
pd.DataFrame({'d': dr}).to_csv(f'{DIR}/downwind_placebo_draws.csv',
                               index=False)
