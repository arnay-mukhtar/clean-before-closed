# Hourly downwind design: which exit events have an HOURLY PM2.5 monitor
# within 25 km in the years before and after? Uses EPA's annual monitor
# listing and reads only site, year, sample duration and observation
# counts (no concentration values). Then recomputes power with the real
# number of covered events.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, zipfile, requests
import numpy as np, pandas as pd
from math import erf
Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

p = f'{DIR}/aqs_hourly_monitor_years.csv'
if not os.path.exists(p):
    parts = []
    for y in range(2004, 2025):
        r = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/'
                         f'annual_conc_by_monitor_{y}.zip', timeout=600)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False,
                        usecols=['State Code', 'County Code', 'Site Num',
                                 'Parameter Code', 'Sample Duration',
                                 'Observation Count', 'Latitude',
                                 'Longitude', 'Year'])
        d = d[(d['Parameter Code'] == 88101)
              & d['Sample Duration'].astype(str).str.contains('1 HOUR')]
        d['site'] = (d['State Code'].astype(str) + '-' + d['County Code']
                     .astype(str) + '-' + d['Site Num'].astype(str))
        parts.append(d.groupby(['site', 'Year']).agg(
            n=('Observation Count', 'max'), lat=('Latitude', 'first'),
            lon=('Longitude', 'first')).reset_index())
        print(y, 'hourly PM2.5 monitors', parts[-1]['site'].nunique(),
              flush=True)
    pd.concat(parts).to_csv(p, index=False)
h = pd.read_csv(p)
h = h[h['n'] >= 2000]                 # at least 2,000 valid hours a year
yrs = h.groupby('site')['Year'].apply(set)
st = h.groupby('site')[['lat', 'lon']].first()
Sx = st[['lat', 'lon']].values

P = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
P = P.merge(E[['fid', 'event_year', 'lat', 'lon']], on=['fid', 'event_year'])
P['hourly_monitors'] = [
    sum(1 for s in st.index[km(Sx[:, 0], Sx[:, 1], r.lat, r.lon) <= 25]
        if {r.event_year - 1, r.event_year + 1} <= yrs.get(s, set()))
    for r in P.itertuples()]
P[['fid', 'name', 'event_year', 'so2_drop', 'pred_main',
   'hourly_monitors']].to_csv(f'{DIR}/hourly_event_coverage.csv',
                              index=False)
pl = pd.read_csv(f'{DIR}/downwind_hourly_placebo_draws.csv')
SD, SHARE, LVL = pl['d'].std(), 0.26, 9.2
print(f'\nHourly placebo SD {SD:.4f} log; share of downwind hours {SHARE}; '
      f'level {LVL} µg/m³')
for t in [1000, 2500, 5000, 10000]:
    e = P[(P['so2_drop'] >= t) & (P['hourly_monitors'] > 0)
          & P['pred_main'].notna()]
    if len(e) < 3:
        print(f'SO2 drop >= {t:,}: {len(e)} covered events'); continue
    se = SD / np.sqrt(len(e)) * LVL
    pred = e['pred_main'].mean() / SHARE
    z = pred / se
    print(f'SO2 drop >= {t:,} t: {len(e)} events with an hourly monitor '
          f'(years {e.event_year.min()}-{e.event_year.max()}); downwind '
          f'prediction {pred:.2f} µg/m³; SE {se:.3f}; {z:.1f} SE; power '
          f'about {100 * (Phi(z - 1.96) + Phi(-z - 1.96)):.0f}%')
