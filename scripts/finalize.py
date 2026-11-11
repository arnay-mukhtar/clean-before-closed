# Freeze the final confirmatory designs (no outcomes near exit plants):
#  PM2.5: near monitors within 35 km, primary sample SO2 drop >= 5,000 t;
#  sulfate: near monitors within 100 km (50 km robustness), same threshold.
# Saves per-event predictions (main/high/low layer, by component), the
# placebo draws for each design, and SHA-256 fingerprints.
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


def event_preds(q, st, Sx, yrs, bgm, lvl, radius, species):
    out = []
    for r in E.itertuples():
        if r.cell not in rc:
            continue
        j = rc[r.cell]; b, a_ = r.event_year - 1, r.event_year + 1
        d = km(Sx[:, 0], Sx[:, 1], r.lat, r.lon)
        near = [s for s in st.index[d <= radius]
                if {b, a_} <= yrs.get(s, set()) and st.at[s, 'cell'] >= 0]
        pairs = []
        for s in near:
            bg = [x for x in st.index[(st['state'] == st.at[s, 'state'])
                                      & st['bg'] & (st['cell'] >= 0)]
                  if {b, a_} <= yrs.get(x, set())]
            if bg:
                pairs.append((st.at[s, 'cell'], st.loc[bg, 'cell'].values,
                              st.at[s, 'state']))
        o = dict(fid=r.fid, name=r.name, event_year=r.event_year,
                 so2_drop=r.so2_drop, nox_drop=r.nox_drop,
                 pm25_drop_tpy=r.pm25_drop_tpy, layer=r.layer,
                 n_near=len(near), n_near_bg=len(pairs))
        for L, tag in [(r.layer, 'main'), (2, 'high'), (1, 'low')]:
            comps = {'so4': R[f'pSO4_L{L}'][j].astype('f8') * max(r.so2_drop, 0)}
            if species == 'pm':
                comps['no3'] = R[f'pNO3_L{L}'][j].astype('f8') * max(r.nox_drop, 0)
                comps['ppm'] = R[f'PrimaryPM25_L{L}'][j].astype('f8') \
                    * r.pm25_drop_tpy
            tot = 0.0
            for k, f in comps.items():
                v = [(f[mc] - f[bc].mean()) * TPY for mc, bc, _ in pairs]
                o[f'{k}_{tag}'] = np.mean(v) if v else np.nan
                tot = tot + o[f'{k}_{tag}']
            o[f'pred_{tag}'] = tot
        if species == 'so4' and pairs:
            lv = lvl.get((pairs[0][2], b), np.nan)
            o['bg_level'] = lv
            for tag in ['main', 'high', 'low']:
                o[f'pred_rel_{tag}'] = o[f'pred_{tag}'] / lv
        out.append(o)
    return pd.DataFrame(out)

def placebo(q, st, Sx, yrs, bgm, lvl, radius, years, val):
    qq = q.join(bgm, on=['state', 'ym']).dropna(subset=['bgl'])
    qq['rel'] = qq['l'] - qq['bgl']
    ann = qq.groupby(['site', 'yr']).agg(rel=('rel', 'mean'),
                                         n=('rel', 'size'),
                                         lv=(val, 'mean'))
    ann = ann[ann['n'] >= 8]
    allowed = set(st.index[st['d_exit'] > radius])
    D, LV = {}, {}
    for c in ctrl.itertuples():
        d = km(Sx[:, 0], Sx[:, 1], c.latitude, c.longitude)
        near = [s for s in st.index[d <= radius] if s in allowed]
        for y in range(2005, 2019):
            v, l = [], []
            for s in near:
                if (s, y - 1) in ann.index and (s, y + 1) in ann.index:
                    v.append(ann.at[(s, y - 1), 'rel'] - ann.at[(s, y + 1), 'rel'])
                    l.append(ann.at[(s, y - 1), 'lv'])
            if v:
                D[(c.facilityid, y)], LV[(c.facilityid, y)] = np.mean(v), np.mean(l)
    plants = sorted({f for f, _ in D})
    rng = np.random.default_rng(20261109)
    dr, lv = [], []
    for _ in range(1000):
        for f in plants:
            y = int(rng.choice(years))
            if (f, y) in D:
                dr.append(D[(f, y)]); lv.append(LV[(f, y)])
    return pd.DataFrame({'d': dr, 'pre_level': lv}), len(plants)

import hashlib
fp = []
def save(df, name):
    p = f'{DIR}/{name}'
    df.to_csv(p, index=False)
    fp.append((name, hashlib.sha256(open(p, 'rb').read()).hexdigest()))

q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
pmq = prep(q, 'pm', 8)
PM = event_preds(*pmq, 35, 'pm')
save(PM, 'final_pm25_35km_predictions.csv')
prim = PM[(PM['n_near_bg'] > 0) & (PM['so2_drop'] >= 5000)]
dr, npl = placebo(*pmq, 35, prim['event_year'].values, 'pm')
save(dr, 'final_pm25_35km_placebo.csv')
lvl = dr['pre_level'].mean()
se = dr['d'].std() / np.sqrt(len(prim)) * lvl
m = prim['pred_main'].mean()
print(f'PM2.5, 35 km, SO2 drop >= 5,000 t: {len(prim)} events; mean main '
      f'prediction {m:.3f} µg/m³ (sulfate {prim.so4_main.mean():.3f}, nitrate '
      f'{prim.no3_main.mean():.3f}, primary {prim.ppm_main.mean():.3f}); '
      f'placebo SD {dr.d.std():.4f} from {npl} plants (mean {dr.d.mean():.4f});'
      f' level {lvl:.1f}; SE {se:.3f} µg/m³; 1.1 is {(1.1 - m) / se:.1f} SE '
      f'above the prediction (power {pw1((1.1 - m) / se):.2f}); zero test '
      f'{m / se:.1f} SE')

q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
soq = prep(q, 'so4', 3)
for rad in [100, 50]:
    SO = event_preds(*soq, rad, 'so4')
    save(SO, f'final_so4_{rad}km_predictions.csv')
    pr = SO[(SO['n_near_bg'] > 0) & (SO['so2_drop'] >= 5000)
            & SO['pred_rel_main'].notna()]
    dr, npl = placebo(*soq, rad, pr['event_year'].values, 'so4')
    save(dr, f'final_so4_{rad}km_placebo.csv')
    se = dr['d'].std() / np.sqrt(len(pr))
    m = pr['pred_rel_main'].mean()
    print(f'Sulfate, {rad} km, SO2 drop >= 5,000 t: {len(pr)} events; mean '
          f'predicted relative fall {100 * m:.2f}% (main layer); placebo SD '
          f'{dr.d.std():.4f} from {npl} plants (mean {dr.d.mean():.4f}); SE '
          f'{100 * se:.2f}%; {m / se:.1f} SE; power {pw2(m / se):.2f}')

print('\nFINGERPRINTS')
for n, h in fp:
    print(f'{n}  {h}')
