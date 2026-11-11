DIR = '/content/drive/MyDrive/coal_closures'
import requests, time, math, os, numpy as np, pandas as pd
from google.colab import userdata
KEY = userdata.get('OPENAQ_KEY')
H = {'X-API-Key': KEY}
API = 'https://api.openaq.org/v3'
ctrl = pd.read_csv(f'{DIR}/us_operating_controls.csv')
CACHE = f'{DIR}/controls_pm25_monthly.csv'

def get(path, params):
    for a in range(8):
        try:
            r = requests.get(API + path, params=params,
                             headers=H, timeout=90)
        except requests.exceptions.RequestException:
            time.sleep(10 * (a + 1)); continue
        if r.status_code == 429:
            time.sleep(20 * (a + 1)); continue
        return r
    return None

def hav(a, b, c, d):
    a, b, c, d = map(math.radians, [a, b, c, d])
    x = (math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c)
         * math.sin((d - b) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(x))

def pm_sensors(lat, lon, rad):
    r = get('/locations', {'coordinates': f'{lat:.5f},{lon:.5f}',
                           'radius': rad, 'parameters_id': 2,
                           'limit': 1000})
    out = []
    if r is None or not r.ok:
        return out
    for L in r.json().get('results', []):
        c = L.get('coordinates') or {}
        if c.get('latitude') is None:
            continue
        d = hav(lat, lon, c['latitude'], c['longitude'])
        f = ((L.get('datetimeFirst') or {}).get('utc') or '')[:4]
        l = ((L.get('datetimeLast') or {}).get('utc') or '')[:4]
        for s in L.get('sensors', []):
            if (s.get('parameter') or {}).get('name') == 'pm25':
                out.append(dict(sid=s['id'], dist=d, first=f, last=l))
    return out

def monthly(sid):
    r = get(f'/sensors/{sid}/days/monthly',
            {'datetime_from': '2016-01-01', 'datetime_to': '2026-01-01',
             'limit': 1000})
    if r is None or not r.ok:
        return pd.DataFrame()
    rows = []
    for x in r.json().get('results', []):
        t = ((x.get('period') or {}).get('datetimeFrom') or {}).get('utc')
        if t and x.get('value') is not None:
            rows.append(dict(ym=int(t[:4] + t[5:7]), pm=x['value']))
    return pd.DataFrame(rows)

done = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
have = set(done['pid']) if len(done) else set()
store = [done] if len(done) else []
for i, p in ctrl.iterrows():
    if p.plant_id in have:
        continue
    ss = pm_sensors(p.lat, p.lon, 25000)
    time.sleep(1.1)
    ss = [s for s in ss if s['first'] and s['first'] <= '2019'
          and s['last'] >= '2023']
    near = sorted([s for s in ss if s['dist'] <= 10],
                  key=lambda s: s['dist'])[:3]
    far = sorted([s for s in ss if 15 <= s['dist'] <= 25],
                 key=lambda s: s['dist'])[:3]
    for ring, lst in [('near', near), ('far', far)]:
        for s in lst:
            m = monthly(s['sid'])
            time.sleep(1.1)
            if len(m):
                m['pid'] = p.plant_id; m['ring'] = ring
                m['sid'] = s['sid']
                store.append(m)
    if i % 10 == 0 and store:
        pd.concat(store).to_csv(CACHE, index=False)
        print(i, p['name'], 'near', len(near), 'far', len(far))
mon = pd.concat(store)
mon.to_csv(CACHE, index=False)

g = mon[mon['pm'] > 0].groupby(['pid', 'ym', 'ring'])['pm'].mean()
w = g.unstack('ring').dropna().reset_index()
w['ratio'] = np.log(w['near']) - np.log(w['far'])
w['per'] = pd.PeriodIndex(w['ym'].astype(str), freq='M')
rng = np.random.default_rng(20261109)
months = pd.period_range('2018-06', '2024-12', freq='M')

def pdelta(g, e):
    pre = g[(g['per'] >= e - 12) & (g['per'] < e)]['ratio']
    post = g[(g['per'] > e) & (g['per'] <= e + 12)]['ratio']
    if len(pre) < 8 or len(post) < 8:
        return np.nan
    return post.mean() - pre.mean()

groups = {k: v for k, v in w.groupby('pid')}
D = np.array([[pdelta(v, months[rng.integers(len(months))])
               for v in groups.values()] for _ in range(300)])
sd = np.nanstd(D)
n_ok = int(np.mean(np.sum(~np.isnan(D), 1)))
print('\nControl plants with near and far PM2.5 monitors:', n_ok)
print('SD of one plant placebo change (log points):', round(sd, 3))
for n in [20, 34, 60]:
    mde = 2.8 * sd / np.sqrt(n)
    print(f'N={n}: MDE = {mde:.3f} log points'
          f' ({100 * (1 - np.exp(-mde)):.1f}% fall in near/far PM2.5)')
