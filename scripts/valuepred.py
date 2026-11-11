# Step: (A) health damages from exiting coal units, before vs at retirement;
# (B) InMAP-predicted PM2.5 fall at monitors for all 540 plant exit events.
# Uses emissions, model output, monitor locations and record presence only.
DIR = '/content/drive/MyDrive/coal_closures'
import hashlib
import numpy as np, pandas as pd

ST2T = 0.90718474                       # short ton -> metric tonne
TPY = 907184.74e6 / (365.25 * 86400)    # µg/s per short ton per year

fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
# check: marginal-values row i is grid cell i
G = np.load(f'{DIR}/isrm_grid.npz')
if not (np.allclose(mv['Location_W'].values, G['W'])
        and np.allclose(mv['Location_S'].values, G['S'])):
    print('Reordering marginal values to the matrix grid')
    key = pd.Series(range(len(mv)), index=pd.MultiIndex.from_arrays(
        [mv['Location_W'].round(0), mv['Location_S'].round(0)]))
    idx = key.reindex(pd.MultiIndex.from_arrays(
        [np.round(G['W'], 0), np.round(G['S'], 0)])).values
    assert not np.isnan(idx).any(), 'grid mismatch'
    mv = mv.iloc[idx.astype(int)].reset_index(drop=True)
cellmap = fc.set_index('facilityid')['cell']

# ---------- (A) damages before vs at retirement ----------
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)
U['cell'] = U['fid'].map(cellmap).fillna(-1).astype(int)
U = U[U['cell'] >= 0].copy()
out = {}
for h in ['high', 'low']:
    md_s = mv[f'MD_SO2_{h}'].values[U['cell']]
    md_n = mv[f'MD_NOx_{h}'].values[U['cell']]
    mm_s = mv[f'MM_SO2_{h}'].values[U['cell']]
    mm_n = mv[f'MM_NOx_{h}'].values[U['cell']]
    for base in ['t10', 'peak']:
        b_d = (U[f'so2_{base}'] * md_s + U[f'nox_{base}'] * md_n) * ST2T
        f_d = (U['so2_final'] * md_s + U['nox_final'] * md_n) * ST2T
        b_m = (U[f'so2_{base}'] * mm_s + U[f'nox_{base}'] * mm_n) * ST2T
        f_m = (U['so2_final'] * mm_s + U['nox_final'] * mm_n) * ST2T
        ok = b_d > 0
        out[(h, base)] = dict(
            share_before=1 - f_d[ok].sum() / b_d[ok].sum(),
            dmg_base_bn=b_d[ok].sum() / 1e9, dmg_final_bn=f_d[ok].sum() / 1e9,
            deaths_base=b_m[ok].sum(), deaths_final=f_m[ok].sum(),
            overstatement=b_d[ok].sum() / f_d[ok].sum())
    U[f'dmg_final_{h}'] = (U['so2_final'] * md_s
                           + U['nox_final'] * md_n) * ST2T
U.to_csv(f'{DIR}/us_unit_exits_damages.csv', index=False)

print(f'(A) {len(U)} coal unit exits 2005-2024, InMAP damages '
      '(per year, summed over units)')
for (h, base), r in out.items():
    print(f'  {h} stack, baseline {base}: share of damage cut before the '
          f'last full year {100 * r["share_before"]:.0f}%; '
          f'baseline ${r["dmg_base_bn"]:.0f}bn and {r["deaths_base"]:,.0f}'
          f' deaths/yr -> last full year ${r["dmg_final_bn"]:.0f}bn and '
          f'{r["deaths_final"]:,.0f} deaths/yr; a baseline-emissions '
          f'valuation overstates the retirement itself by '
          f'{r["overstatement"]:.1f}x')

# ---------- (B) predicted fall at monitors for plant events ----------
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
E['cell'] = E['fid'].map(cellmap).fillna(-1).astype(int)
R = np.load(f'{DIR}/isrm_rows_allcoal.npz')
rcells = list(R['cells'])
rows = {k: R[k] for k in R.files if k != 'cells'}

frames = [pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                      usecols=['site', 'ym', 'lat', 'lon', 'state', 'ndays'])
          for y in range(2000, 2026)]
q = pd.concat(frames)
q = q[q['ndays'] >= 8]
q['yr'] = q['ym'] // 100
sites = q.groupby('site')[['lat', 'lon', 'state']].first()
n_yr = q.groupby(['site', 'yr']).size()
yrs = n_yr[n_yr >= 8].reset_index().groupby('site')['yr'].apply(set)

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

from pyproj import Transformer
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, Ee, S, N = G['W'], G['E'], G['S'], G['N']
def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    h = np.where((W <= x) & (x < Ee) & (S <= y) & (y < N))[0]
    return int(h[0]) if len(h) else -1
sites['cell'] = [cell_of(a, o) for a, o in sites[['lat', 'lon']].values]
Sx = sites[['lat', 'lon']].values
mind = np.full(len(Sx), np.inf)
for la, lo in fc[['latitude', 'longitude']].values:
    mind = np.minimum(mind, km(Sx[:, 0], Sx[:, 1], la, lo))
sites['bg'] = (mind > 50) & (sites['cell'] >= 0)

res = []
for r in E.itertuples():
    if r.cell < 0 or r.cell not in rcells:
        continue
    j = rcells.index(r.cell)
    b, a_ = r.event_year - 1, r.event_year + 1
    d = km(Sx[:, 0], Sx[:, 1], r.lat, r.lon)
    near = [s for s in sites.index[d <= 25]
            if {b, a_} <= yrs.get(s, set()) and sites.at[s, 'cell'] >= 0]
    rec = dict(fid=r.fid, name=r.name, state=r.state,
               event_year=r.event_year, so2_drop=r.so2_drop,
               nox_drop=r.nox_drop, n_near=len(near))
    for L, tag in [(2, 'high'), (1, 'low')]:
        fall = (rows[f'pSO4_L{L}'][j].astype('f8') * max(r.so2_drop, 0)
                + rows[f'pNO3_L{L}'][j].astype('f8') * max(r.nox_drop, 0)
                ) * TPY
        vals = []
        for s in near:
            st = sites.at[s, 'state']
            bg = sites.index[(sites['state'] == st) & sites['bg']]
            bg = [x for x in bg if {b, a_} <= yrs.get(x, set())]
            if not bg:
                continue
            vals.append(fall[sites.at[s, 'cell']]
                        - fall[sites.loc[bg, 'cell'].values].mean())
        rec[f'n_near_bg'] = len(vals)
        rec[f'pred_contrast_{tag}'] = np.mean(vals) if vals else np.nan
        rec[f'pred_plantcell_{tag}'] = fall[r.cell]
    res.append(rec)
P = pd.DataFrame(res)
path = f'{DIR}/model_predictions_us_events.csv'
P.to_csv(path, index=False)
h = hashlib.sha256(open(path, 'rb').read()).hexdigest()

pd.set_option('display.width', 220)
m = P[P['n_near_bg'] > 0]
print(f'\n(B) Events with monitor and state background: {len(m)}')
for t in [0, 1000, 5000, 10000, 20000]:
    sub = m[m['so2_drop'] >= t]
    print(f'  SO2 drop >= {t:>6,} t: n={len(sub):3d}; predicted fall near '
          f'minus background, µg/m³: mean {sub.pred_contrast_high.mean():.3f}'
          f' (high), {sub.pred_contrast_low.mean():.3f} (low); median '
          f'{sub.pred_contrast_high.median():.3f} (high)')
print('\nLargest predicted falls (high stack):')
print(m.sort_values('pred_contrast_high', ascending=False)
      [['name', 'state', 'event_year', 'so2_drop', 'nox_drop', 'n_near_bg',
        'pred_contrast_high', 'pred_contrast_low']].round(3).head(20)
      .to_string(index=False))
print('\nSHA-256 of model_predictions_us_events.csv:', h)
