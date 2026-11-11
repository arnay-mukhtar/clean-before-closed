# How many CLOSED US plants have an AQS PM2.5 monitor within 25 km with
# records on both sides of the closure? Uses monitor locations and record
# presence only; no PM2.5 values are inspected.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd
frames = [pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                      usecols=['site', 'ym', 'lat', 'lon', 'ndays'])
          for y in range(2016, 2026)]
a = pd.concat(frames)
a = a[a['ndays'] >= 8]
sites = a.groupby('site')[['lat', 'lon']].first()
months_by_site = a.groupby('site')['ym'].apply(set)

plants = pd.read_csv(f'{DIR}/plants.csv')
dates = pd.read_csv(f'{DIR}/us_closure_dates.csv')
dates = dates[dates['clean_closure']]
us = plants.merge(dates[['plant_id', 'event_ym']], on='plant_id')

def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = (np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2)
         * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

S = sites.values
out = []
for _, p in us.iterrows():
    e = pd.Period(str(int(p['event_ym'])), 'M')
    pre = {int((e - k).strftime('%Y%m')) for k in range(1, 13)}
    post = {int((e + k).strftime('%Y%m')) for k in range(1, 13)}
    d = km(S[:, 0], S[:, 1], p['lat'], p['lon'])
    n_ok = 0
    for s in sites.index[d <= 25]:
        have = months_by_site[s]
        if len(have & pre) >= 8 and len(have & post) >= 8:
            n_ok += 1
    out.append(dict(name=p['name'], event_ym=int(p['event_ym']),
                    monitors_ok=n_ok))
o = pd.DataFrame(out)
o.to_csv(f'{DIR}/closed_aqs_coverage.csv', index=False)
print('Clean US closures:', len(o))
print('With >=1 AQS monitor within 25 km covering 12m before and after:',
      (o['monitors_ok'] > 0).sum())
