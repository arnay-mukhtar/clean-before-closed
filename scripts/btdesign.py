# POST-REGISTRATION, exploratory: reconciling with Brown and Tousey (2020).
# Their 1.1-1.4 ug/m3 is the coefficient on the MONTH of closure in a monthly
# monitor panel with monitor-by-year and calendar-month fixed effects
# (Eq. 1: Pre, closure-month and months-since-closure terms; post-closure
# months are the reference; standard errors clustered by monitor-year;
# monitors within 25 or 50 miles of a closing plant; a monitor's data are
# kept until the month before its next closure). Here:
#  (1) their specification on our data (closures 2001-2018, data 2000-2018)
#  (2) the same with year-by-month fixed effects instead of calendar-month
#      (absorbs shocks common to all monitors in a given month)
#  (3) the lasting effect: post-closure indicator with monitor and
#      year-by-month fixed effects, 36 months either side
#  (4) event-time coefficients by quarter, -8..+8 (figure data)
# Closure month: first month without coal burning at a plant whose coal
# units all stop, from EPA monthly unit data (fetched once, cached).
DIR = '/content/drive/MyDrive/coal_closures'
import os, time, requests
import numpy as np, pandas as pd

OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
MILE = 1.609344

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

# ---------- plant closures (whole-plant coal exits) ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
cy = a[a['coal']].groupby('facilityid')['year'].agg(['min', 'max', 'nunique'])
YMAX = int(a['year'].max())
clo = cy[(cy['max'] < YMAX) & (cy['nunique'] >= 2) & (cy['max'] >= 2000)]
coal_units = a[a['coal']].groupby('facilityid')['unitid'] \
    .apply(lambda s: set(s.astype(str)))
print(f'{len(clo)} plants stop burning coal between 2000 and {YMAX - 1}')

p_m = f'{DIR}/us_epa_monthly_closures.csv'
URL = ('https://api.epa.gov/easey/streaming-services/emissions/'
       'apportioned/monthly')
if os.path.exists(p_m):
    mon = pd.read_csv(p_m)
else:
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    parts = []
    for i, (fid, r) in enumerate(clo.iterrows()):
        for tries in range(6):
            q = requests.get(URL, params=dict(
                api_key=key, facilityId=int(fid),
                year='|'.join(str(y) for y in [r['max'], r['max'] + 1]),
                month='|'.join(str(m) for m in range(1, 13))), timeout=180)
            if q.status_code == 429:
                time.sleep(10 * (tries + 1)); continue
            q.raise_for_status(); break
        d = pd.DataFrame(q.json())
        if len(d):
            d.columns = [c.lower() for c in d.columns]
            parts.append(d[['facilityid', 'unitid', 'year', 'month',
                            'heatinput']])
        if (i + 1) % 25 == 0:
            print(f'  monthly data: {i + 1} of {len(clo)} plants', flush=True)
    mon = pd.concat(parts); mon.to_csv(p_m, index=False)
mon['heatinput'] = pd.to_numeric(mon['heatinput'], errors='coerce').fillna(0)
mon = mon[mon.apply(lambda r: str(r['unitid']) in
                    coal_units.get(int(r['facilityid']), set()), axis=1)]
lastm = mon[mon['heatinput'] > 0].assign(
    ym=lambda d: d['year'].astype(int) * 12 + d['month'].astype(int) - 1) \
    .groupby('facilityid')['ym'].max()
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
    .dropna(subset=['latitude', 'longitude'])
C = pd.DataFrame({'fid': clo.index})
C['close_m'] = C['fid'].map(lastm) + 1          # first month without coal
miss = C['close_m'].isna()
C.loc[miss, 'close_m'] = (clo.loc[C.loc[miss, 'fid'], 'max'].values + 1) * 12
C = C.merge(fc[['facilityid', 'latitude', 'longitude']], left_on='fid',
            right_on='facilityid')
print(f'{len(C)} closures located; {miss.sum()} dated to January after the '
      'last coal year (no monthly record)')

# ---------- monitor panel ----------
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)
               if os.path.exists(f'{DIR}/aqs/pm25_monthly_{y}.csv')])
q = q[(q['ndays'] >= 4) & (q['pm'] > 0)].copy()
q['m'] = (q['ym'] // 100) * 12 + q['ym'] % 100 - 1   # month index
q['yr'] = q['ym'] // 100; q['moy'] = q['ym'] % 100
st = q.groupby('site')[['lat', 'lon']].first()

def demean(Y, groups, it=500, tol=1e-10):
    Y = Y.copy()
    for _ in range(it):
        old = Y.copy()
        for g, n in groups:
            Y -= (np.stack([np.bincount(g, Y[:, j], minlength=n)
                            for j in range(Y.shape[1])], 1)
                  / np.bincount(g, minlength=n)[:, None])[g]
        if np.abs(Y - old).max() < tol * (1 + np.abs(old).max()):
            break
    return Y

def ols_fe(d, y, xs, fes, cl):
    groups = [(pd.factorize(d[f])[0], d[f].nunique()) for f in fes]
    Z = demean(d[[y] + xs].values.astype(float), groups)
    yv, X = Z[:, 0], Z[:, 1:]
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ yv; e = yv - X @ b
    g = pd.factorize(d[cl])[0]; G = g.max() + 1
    S = np.stack([np.bincount(g, X[:, j] * e, minlength=G)
                  for j in range(X.shape[1])], 1)
    V = XtX @ (S.T @ S) @ XtX * G / (G - 1)
    return pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(V))}, index=xs)

res = []
for miles in [25, 50]:
    r = miles * MILE
    Dm = np.stack([km(st['lat'].values, st['lon'].values, la, lo)
                   for la, lo in fc[['latitude', 'longitude']].values], 1)
    near_any = st.index[(Dm <= r).any(1)]          # monitors near any coal plant
    Dc = np.stack([km(st['lat'].values, st['lon'].values, la, lo)
                   for la, lo in C[['latitude', 'longitude']].values], 1)
    first, second = {}, {}
    for i, s in enumerate(st.index):
        ms = np.sort(C['close_m'].values[Dc[i] <= r])
        if len(ms):
            first[s] = ms[0]
            second[s] = ms[1] if len(ms) > 1 else np.inf
    d = q[q['site'].isin(near_any)].copy()
    d['c1'] = d['site'].map(first); d['c2'] = d['site'].map(second)
    d = d[~(d['m'] >= d['c2'])]                    # keep until next closure
    has = d['c1'].notna()
    d['pre'] = (has & (d['m'] < d['c1'])).astype(float)
    d['cpc'] = (has & (d['m'] == d['c1'])).astype(float)
    d['msc'] = np.where(has & (d['m'] > d['c1']), d['m'] - d['c1'], 0.0)
    d['post'] = (has & (d['m'] >= d['c1'])).astype(float)
    d['site_yr'] = d['site'].astype(str) + '_' + d['yr'].astype(str)
    bt = d[(d['yr'] <= 2018) & (~has | ((d['c1'] >= 2001 * 12)
                                        & (d['c1'] < 2019 * 12)))]
    lab = f'{miles} miles'
    print(f'\n{lab}: {bt.site.nunique()} monitors, '
          f'{int(bt.groupby("site").c1.first().notna().sum())} with a '
          f'closure; mean PM2.5 {bt.pm.mean():.2f}', flush=True)
    for name, dd, xs, fes, cl in [
            ('(1) their specification', bt, ['pre', 'cpc', 'msc'],
             ['site_yr', 'moy'], 'site_yr'),
            ('(2) year-by-month FE', bt, ['pre', 'cpc', 'msc'],
             ['site_yr', 'm'], 'site_yr'),
            ('(3) lasting effect, +-36 months', d[~has | ((d['m'] - d['c1'])
                                                         .abs() <= 36)],
             ['post'], ['site', 'm'], 'site')]:
        o = ols_fe(dd, 'pm', xs, fes, cl)
        for x, rr in o.iterrows():
            res.append(dict(buffer=lab, model=name, term=x, coef=rr.coef,
                            se=rr.se, n=len(dd)))
        print(f'  {name}: ' + '; '.join(f'{x} {rr.coef:+.3f} ({rr.se:.3f})'
                                       for x, rr in o.iterrows()), flush=True)
    # (4) event time by quarter, monitors with a closure, monitor + month FE
    ev = d[has].copy()
    ev['k'] = np.floor((ev['m'] - ev['c1']) / 3).astype(int).clip(-9, 9)
    dums = []
    for k in range(-9, 10):
        if k == -1:
            continue
        ev[f'q{k}'] = (ev['k'] == k).astype(float); dums.append(f'q{k}')
    ctrl = d[~has].copy()
    for c in dums:
        ctrl[c] = 0.0
    o = ols_fe(pd.concat([ev, ctrl]), 'pm', dums, ['site', 'm'], 'site')
    o['buffer'] = lab
    o.to_csv(f'{OUT}/bt_eventtime_{miles}mi.csv')
    print('  (4) quarters from closure (ref: quarter -1; ends binned): '
          + ', '.join(f'{x[1:]}: {rr.coef:+.2f}' for x, rr in o.iterrows()))
pd.DataFrame(res).to_csv(f'{OUT}/bt_design.csv', index=False)
print('\nBrown and Tousey (2020, Table 3): closure month -1.129 (0.466) at '
      '25 miles, -1.401 (0.398) at 50 miles; pre 0.161 (0.351) and -0.104 '
      '(0.308).')
