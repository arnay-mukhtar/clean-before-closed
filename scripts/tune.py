# Pre-filing design tuning (no outcomes near exit plants are read):
# (1) PM2.5: near radius 15/25/35 km x SO2-drop threshold 1,000-10,000 t;
# (2) sulfate: near radius 25/50/100 km x the same thresholds;
#     power from frozen predictions and placebo noise at never-exit plants
#     (monitors farther than the radius from every plant with an exit);
# (3) Europe: air-quality station coverage around large unit exits
#     (station metadata only).
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, requests
import numpy as np, pandas as pd
from math import erf
from pyproj import Transformer

Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))
pw2 = lambda z: Phi(z - 1.96) + Phi(-z - 1.96)       # two-sided 5%
pw1 = lambda z: Phi(z - 1.645)                        # one-sided 5%
TPY = 907184.74e6 / (365.25 * 86400)
THR = [1000, 2500, 5000, 10000]

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
G = np.load(f'{DIR}/isrm_grid.npz')
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, Ee, S, N = G['W'], G['E'], G['S'], G['N']
def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    h = np.where((W <= x) & (x < Ee) & (S <= y) & (y < N))[0]
    return int(h[0]) if len(h) else -1
R = np.load(f'{DIR}/isrm_rows_allcoal.npz')
rc = {c: i for i, c in enumerate(R['cells'])}

P = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
E = E.merge(P[['fid', 'event_year', 'layer', 'pm25_drop_tpy']],
            on=['fid', 'event_year'], how='left')
E['layer'] = E['layer'].fillna(2).astype(int)
E['pm25_drop_tpy'] = E['pm25_drop_tpy'].fillna(0)
E['cell'] = E['fid'].map(fc.set_index('facilityid')['cell']).fillna(-1) \
             .astype(int)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ex = fc[fc['facilityid'].isin(exit_fids)]
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
cy = a[a['primaryfuelinfo'].fillna('').str.contains('Coal')
       & (a['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
ctrl = fc[fc['facilityid'].isin([f for f, ys in cy.items()
                                 if set(range(2003, 2020)) <= ys
                                 and f not in exit_fids])]

def prep(q, val, min_days):
    q = q[(q['ndays'] >= min_days) & (q[val] > 0)].copy()
    q['yr'] = q['ym'] // 100
    st = q.groupby('site')[['lat', 'lon', 'state']].first()
    Sx = st[['lat', 'lon']].values
    dc = np.full(len(Sx), np.inf); de = np.full(len(Sx), np.inf)
    for la, lo in fc[['latitude', 'longitude']].values:
        dc = np.minimum(dc, km(Sx[:, 0], Sx[:, 1], la, lo))
    for la, lo in ex[['latitude', 'longitude']].values:
        de = np.minimum(de, km(Sx[:, 0], Sx[:, 1], la, lo))
    st['bg'] = dc > 50; st['d_exit'] = de
    st['cell'] = [cell_of(x, y) for x, y in Sx]
    ny = q.groupby(['site', 'yr']).size()
    yrs = ny[ny >= 8].reset_index().groupby('site')['yr'].apply(set)
    q['l'] = np.log(q[val])
    bgm = (q[q['site'].isin(set(st.index[st['bg']]))]
           .groupby(['state', 'ym'])['l'].mean().rename('bgl'))
    lvl = (q[q['site'].isin(set(st.index[st['bg']]))]
           .groupby(['state', 'yr'])[val].mean())
    return q, st, Sx, yrs, bgm, lvl

def design(q, st, Sx, yrs, bgm, lvl, radius, species, relative):
    # predictions (no outcomes)
    rows = []
    for r in E.itertuples():
        if r.cell not in rc:
            continue
        j = rc[r.cell]; b, a_ = r.event_year - 1, r.event_year + 1
        d = km(Sx[:, 0], Sx[:, 1], r.lat, r.lon)
        near = [s for s in st.index[d <= radius]
                if {b, a_} <= yrs.get(s, set()) and st.at[s, 'cell'] >= 0]
        L = r.layer
        f = R[f'pSO4_L{L}'][j].astype('f8') * max(r.so2_drop, 0)
        if species == 'pm':
            f = f + R[f'pNO3_L{L}'][j].astype('f8') * max(r.nox_drop, 0) \
                + R[f'PrimaryPM25_L{L}'][j].astype('f8') * r.pm25_drop_tpy
        f = f * TPY
        vals, sts = [], []
        for s in near:
            bg = [x for x in st.index[(st['state'] == st.at[s, 'state'])
                                      & st['bg'] & (st['cell'] >= 0)]
                  if {b, a_} <= yrs.get(x, set())]
            if bg:
                vals.append(f[st.at[s, 'cell']]
                            - f[st.loc[bg, 'cell'].values].mean())
                sts.append(st.at[s, 'state'])
        if vals:
            p = np.mean(vals)
            if relative:
                lv = lvl.get((sts[0], b), np.nan)
                p = p / lv if lv and np.isfinite(lv) else np.nan
            rows.append((r.so2_drop, p, r.event_year))
    ev = pd.DataFrame(rows, columns=['so2_drop', 'pred', 'year']).dropna()
    # placebo at never-exit plants, monitors farther than radius from exits
    qq = q.join(bgm, on=['state', 'ym']).dropna(subset=['bgl'])
    qq['rel'] = qq['l'] - qq['bgl']
    ann = qq.groupby(['site', 'yr']).agg(rel=('rel', 'mean'),
                                         n=('rel', 'size'))
    ann = ann[ann['n'] >= 8]
    allowed = set(st.index[st['d_exit'] > radius])
    D = []
    for c in ctrl.itertuples():
        d = km(Sx[:, 0], Sx[:, 1], c.latitude, c.longitude)
        near = [s for s in st.index[d <= radius] if s in allowed]
        for y in range(2005, 2019):
            v = [ann.at[(s, y - 1), 'rel'] - ann.at[(s, y + 1), 'rel']
                 for s in near
                 if (s, y - 1) in ann.index and (s, y + 1) in ann.index]
            if v:
                D.append(np.mean(v))
    return ev, np.std(D) if D else np.nan, len(D)

out = []
# (1) PM2.5
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
pmq = prep(q, 'pm', 8)
LVL_PM = 9.3
for rad in [15, 25, 35]:
    ev, sd, nd = design(*pmq, rad, 'pm', False)
    for t in THR:
        e = ev[ev['so2_drop'] >= t]
        if len(e) < 5:
            continue
        se = sd / np.sqrt(len(e)) * LVL_PM
        m = e['pred'].mean()
        out.append(dict(outcome='PM2.5', radius_km=rad, threshold=t,
                        events=len(e), mean_pred=round(m, 3),
                        se=round(se, 3), z_zero=round(m / se, 2),
                        power_zero=round(pw2(m / se), 2),
                        power_vs_1_1=round(pw1((1.1 - m) / se), 2),
                        placebo_n=nd))
    print('PM2.5 radius', rad, 'done', flush=True)
# (2) sulfate
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
soq = prep(q, 'so4', 3)
for rad in [25, 50, 100]:
    ev, sd, nd = design(*soq, rad, 'so4', True)
    for t in THR:
        e = ev[ev['so2_drop'] >= t]
        if len(e) < 5:
            continue
        se = sd / np.sqrt(len(e))
        m = e['pred'].mean()
        out.append(dict(outcome='sulfate (relative)', radius_km=rad,
                        threshold=t, events=len(e), mean_pred=round(m, 4),
                        se=round(se, 4), z_zero=round(m / se, 2),
                        power_zero=round(pw2(m / se), 2), power_vs_1_1=None,
                        placebo_n=nd))
    print('Sulfate radius', rad, 'done', flush=True)
T = pd.DataFrame(out)
T.to_csv(f'{DIR}/design_tuning.csv', index=False)
pd.set_option('display.width', 200)
print('\nDESIGN TUNING (power from frozen predictions and control-plant '
      'noise only)')
print(T.to_string(index=False))

# (3) Europe: station coverage
print('\nEUROPE: station metadata')
p_meta = f'{DIR}/eea/eea_station_metadata.csv'
if not os.path.exists(p_meta):
    try:
        r = requests.get('https://discomap.eea.europa.eu/map/fme/metadata/'
                         'PanEuropean_metadata.csv', timeout=300)
        print('metadata status', r.status_code, len(r.content))
        if r.ok:
            open(p_meta, 'wb').write(r.content)
    except Exception as ex_:
        print('Could not download station metadata:', str(ex_)[:120])
if os.path.exists(p_meta):
    md = pd.read_csv(p_meta, sep=None, engine='python')
    print('Columns:', list(md.columns))
    md.columns = [c.lower() for c in md.columns]
    pc = md['airpollutantcode'].astype(str).str.extract(r'(\d+)$')[0]
    md['pol'] = pd.to_numeric(pc, errors='coerce')
    md['b'] = pd.to_datetime(md['observationdatebegin'], errors='coerce',
                             utc=True)
    md['e'] = pd.to_datetime(md['observationdateend'], errors='coerce',
                             utc=True)
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
        t0 = pd.Timestamp(f'{r.last_coal_year - 1}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{r.last_coal_year + 1}-12-31', tz='UTC')
        d = km(md['latitude'].values, md['longitude'].values,
               r.Latitude, r.Longitude)
        ok = (d <= 25) & (md['b'] <= t0) & (md['e'].isna() | (md['e'] >= t1))
        rows.append(dict(name=r.name[:40], country=r.country,
                         year=r.last_coal_year, so2_drop=round(r.so2_drop),
                         nox_drop=round(r.nox_drop),
                         pm25_stations=md.loc[ok & (md['pol'] == 6001),
                                              'airqualitystation'].nunique(),
                         so2_stations=md.loc[ok & (md['pol'] == 1),
                                             'airqualitystation'].nunique(),
                         no2_stations=md.loc[ok & (md['pol'] == 8),
                                             'airqualitystation'].nunique()))
    C = pd.DataFrame(rows)
    C.to_csv(f'{DIR}/eu_station_coverage.csv', index=False)
    print(C.to_string(index=False))
    print('Events with a PM2.5 station within 25 km covering both years:',
          int((C['pm25_stations'] > 0).sum()), 'of', len(C))
