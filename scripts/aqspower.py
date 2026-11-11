# PM2.5 power check using official EPA AQS daily data (all US regulatory monitors)
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, zipfile, requests, numpy as np, pandas as pd
RAW = f'{DIR}/aqs'
os.makedirs(RAW, exist_ok=True)
BASE = 'https://aqs.epa.gov/aqsweb/airdata/daily_88101_{}.zip'

frames = []
for y in range(2016, 2026):
    path = f'{RAW}/pm25_monthly_{y}.csv'
    if os.path.exists(path):
        frames.append(pd.read_csv(path)); continue
    r = requests.get(BASE.format(y), timeout=600)
    if not r.ok:
        print(y, 'not available', r.status_code); continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False,
                    usecols=['State Code', 'County Code', 'Site Num',
                             'Latitude', 'Longitude', 'Date Local',
                             'Arithmetic Mean'])
    d['site'] = (d['State Code'].astype(str) + '-'
                 + d['County Code'].astype(str) + '-'
                 + d['Site Num'].astype(str))
    d['ym'] = d['Date Local'].str[:7].str.replace('-', '').astype(int)
    m = (d.groupby(['site', 'ym'])
          .agg(lat=('Latitude', 'first'), lon=('Longitude', 'first'),
               state=('State Code', 'first'),
               pm=('Arithmetic Mean', 'mean'),
               ndays=('Arithmetic Mean', 'size'))
          .reset_index())
    m.to_csv(path, index=False)
    frames.append(m)
    print(y, 'sites', m['site'].nunique())
aqs = pd.concat(frames)
aqs = aqs[(aqs['ndays'] >= 8) & (aqs['pm'] > 0)]

ctrl = pd.read_csv(f'{DIR}/us_operating_controls.csv')
plants = pd.read_csv(f'{DIR}/plants.csv')
coal = pd.concat([ctrl[['lat', 'lon']],
                  plants[plants['country'] == 'United States'][['lat', 'lon']]])

sites = aqs.groupby('site')[['lat', 'lon']].first()
def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2)
         * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))

S = sites.values
mind = np.full(len(S), np.inf)
for la, lo in coal.values:
    mind = np.minimum(mind, km(S[:, 0], S[:, 1], la, lo))
sites['min_coal_km'] = mind
bg_sites = set(sites.index[sites['min_coal_km'] > 50])

aqs['lpm'] = np.log(aqs['pm'])
bg = (aqs[aqs['site'].isin(bg_sites)]
      .groupby(['state', 'ym'])['lpm'].mean().rename('bg'))
aqs = aqs.join(bg, on=['state', 'ym'])
aqs['rel'] = aqs['lpm'] - aqs['bg']

rows = []
for _, p in ctrl.iterrows():
    d = km(S[:, 0], S[:, 1], p.lat, p.lon)
    near = set(sites.index[d <= 25])
    if not near:
        continue
    g = aqs[aqs['site'].isin(near)].groupby('ym')['rel'].mean()
    if len(g) >= 36:
        rows.append((p.plant_id, len(near), g))
print('Control plants with an AQS PM2.5 monitor within 25 km:', len(rows))

rng = np.random.default_rng(20261109)
months = pd.period_range('2018-06', '2024-12', freq='M')
def pdelta(g, e):
    per = pd.PeriodIndex(g.index.astype(str), freq='M')
    v = g.values
    pre = v[(per >= e - 12) & (per < e)]
    post = v[(per > e) & (per <= e + 12)]
    if len(pre) < 8 or len(post) < 8:
        return np.nan
    return post.mean() - pre.mean()
D = np.array([[pdelta(g, months[rng.integers(len(months))])
               for _, _, g in rows] for _ in range(300)])
sd = np.nanstd(D)
print('Usable control plants:', int(np.mean(np.sum(~np.isnan(D), 1))))
print('SD of one plant placebo change (log points):', round(sd, 3))
for n in [20, 34, 50]:
    mde = 2.8 * sd / np.sqrt(n)
    print(f'N={n}: MDE = {mde:.3f} log points'
          f' ({100 * (1 - np.exp(-mde)):.1f}% fall in PM2.5 near plant'
          f' relative to the state background)')
