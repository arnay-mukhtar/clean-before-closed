# Step: plant-level coal exit events 2005-2024 with their emissions drop,
# PM2.5 monitor coverage (record presence only), and the InMAP damage file.
# Reads no PM2.5 values.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, time, zipfile, requests
import numpy as np, pandas as pd

a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False)
for c in ['so2mass', 'noxmass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)

# ---------- facility coordinates (EPA) ----------
p_fac = f'{DIR}/us_epa_facilities.csv'
if os.path.exists(p_fac):
    fac = pd.read_csv(p_fac)
else:
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    parts = []
    for y in range(2000, 2026, 5):
        yrs = '|'.join(str(v) for v in range(y, min(y + 5, 2026)))
        for tries in range(6):
            r = requests.get('https://api.epa.gov/easey/streaming-services/'
                             'facilities/attributes',
                             params=dict(api_key=key, year=yrs), timeout=300)
            if r.status_code == 429:
                time.sleep(15 * (tries + 1)); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json())
        d.columns = [c.lower() for c in d.columns]
        parts.append(d)
        print('facility attributes', yrs, len(d), flush=True)
    fac = pd.concat(parts)
    fac = (fac.dropna(subset=['latitude', 'longitude'])
              .groupby('facilityid')[['latitude', 'longitude', 'statecode',
                                      'facilityname']].first().reset_index())
    fac.to_csv(p_fac, index=False)
fac = fac.set_index('facilityid')

# ---------- plant-level events ----------
py = a.groupby(['facilityid', 'year'])[['so2mass', 'noxmass']].sum()
ev = []
for (fid, yr), g in U.groupby(['fid', 'last_coal_year']):
    try:
        s = py.loc[fid]
    except KeyError:
        continue
    b = s.reindex([yr - 1]).fillna(0).iloc[0]
    af = s.reindex([yr + 1]).fillna(0).iloc[0]
    if fid not in fac.index:
        continue
    f = fac.loc[fid]
    ev.append(dict(fid=fid, name=f['facilityname'], state=f['statecode'],
                   lat=f['latitude'], lon=f['longitude'], event_year=yr,
                   n_units=len(g), so2_before=b['so2mass'],
                   so2_after=af['so2mass'], nox_before=b['noxmass'],
                   nox_after=af['noxmass']))
E = pd.DataFrame(ev)
E['so2_drop'] = E['so2_before'] - E['so2_after']
E['nox_drop'] = E['nox_before'] - E['nox_after']

# ---------- AQS monitor records 2000-2015 (adds to 2016-2025) ----------
RAW = f'{DIR}/aqs'
BASE = 'https://aqs.epa.gov/aqsweb/airdata/daily_88101_{}.zip'
for y in range(2000, 2016):
    path = f'{RAW}/pm25_monthly_{y}.csv'
    if os.path.exists(path):
        continue
    r = requests.get(BASE.format(y), timeout=900)
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
    print(y, 'AQS sites', m['site'].nunique(), flush=True)

frames = [pd.read_csv(f'{RAW}/pm25_monthly_{y}.csv',
                      usecols=['site', 'ym', 'lat', 'lon', 'ndays'])
          for y in range(2000, 2026)
          if os.path.exists(f'{RAW}/pm25_monthly_{y}.csv')]
q = pd.concat(frames)
q = q[q['ndays'] >= 8]
q['yr'] = q['ym'] // 100
sites = q.groupby('site')[['lat', 'lon']].first()
yrs_by_site = q.groupby(['site', 'yr']).size()
yrs_by_site = yrs_by_site[yrs_by_site >= 8].reset_index()
yrs_by_site = yrs_by_site.groupby('site')['yr'].apply(set)

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
S = sites.values
cov = []
for r in E.itertuples():
    d = km(S[:, 0], S[:, 1], r.lat, r.lon)
    n = 0
    for st in sites.index[d <= 25]:
        have = yrs_by_site.get(st, set())
        if (r.event_year - 1) in have and (r.event_year + 1) in have:
            n += 1
    cov.append(n)
E['monitors_25km'] = cov
E.to_csv(f'{DIR}/us_plant_exit_events.csv', index=False)

print(f'\nPlant-level coal exit events 2005-2024: {len(E)}')
for t in [1000, 5000, 10000, 20000]:
    sub = E[E['so2_drop'] >= t]
    print(f'  plant SO2 drop >= {t:,} t: {len(sub)}; with a monitor '
          f'within 25 km (8+ months in the years before and after): '
          f'{int((sub.monitors_25km > 0).sum())}; '
          f'2019+: {int((sub.event_year >= 2019).sum())}')
big = E[(E['so2_drop'] >= 5000) & (E['monitors_25km'] > 0)]
print('  Those with monitors, by event year:',
      big['event_year'].value_counts().sort_index().to_dict())
pd.set_option('display.width', 220)
print('\nLargest plant SO2 drops:')
print(E.sort_values('so2_drop', ascending=False)
      [['name', 'state', 'event_year', 'n_units', 'so2_before', 'so2_after',
        'nox_drop', 'monitors_25km']].round(0).head(25).to_string(index=False))

# ---------- InMAP marginal damages file ----------
p_mv = f'{DIR}/isrm_marginal_values.csv'
if not os.path.exists(p_mv):
    meta = requests.get('https://zenodo.org/api/records/3590127',
                        timeout=60).json()
    files = meta.get('files', [])
    print('\nZenodo files:', [(f.get('key'), f.get('size')) for f in files])
    for f in files:
        if f.get('key', '').endswith('.csv'):
            link = f['links'].get('self') or f['links'].get('download')
            open(p_mv, 'wb').write(requests.get(link, timeout=300).content)
mv = pd.read_csv(p_mv)
print('\nMarginal values file:', mv.shape)
print(list(mv.columns))
print(mv.head(3).T.to_string())
