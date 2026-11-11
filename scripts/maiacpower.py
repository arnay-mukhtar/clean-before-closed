# MAIAC downwind design, power check at never-exit plants (run after the
# exports in MyDrive/maiac_exports have finished).
# Statistic per plant-year: mean daily AOD in the downwind sector minus the
# upwind sector; event change = year before minus year after.
# Expected effect for the events comes from control plants only: the slope
# of the down-minus-up contrast on the plant's SO2 emissions (cross-section
# of control plant-years), times each event's SO2 drop.
DIR = '/content/drive/MyDrive/coal_closures'
import glob
import numpy as np, pandas as pd
from math import erf
Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))

files = sorted(glob.glob('/content/drive/MyDrive/maiac_exports/'
                         'maiac_controls_*.csv'))
print('Export files found:', len(files))
m = pd.concat([pd.read_csv(f) for f in files])
m = m[m['count'] >= 20]                      # at least 20 valid 1 km pixels
w = pd.read_csv(f'{DIR}/wind_daily_plants.csv')
w['to'] = (np.degrees(np.arctan2(w['u'], w['v'])) + 360) % 360
w['down'] = (((w['to'] + 22.5) % 360) // 45).astype(int)
w['up'] = (w['down'] + 4) % 8
x = m.merge(w[['facilityid', 'date', 'down', 'up']],
            left_on=['fid', 'date'], right_on=['facilityid', 'date'])
dn = x[x['sector'] == x['down']][['fid', 'date', 'mean']]
upx = x[x['sector'] == x['up']][['fid', 'date', 'mean']]
d = dn.merge(upx, on=['fid', 'date'], suffixes=('_d', '_u'))
d['du'] = d['mean_d'] - d['mean_u']
d['yr'] = d['date'].str[:4].astype(int)
py = d.groupby(['fid', 'yr']).agg(du=('du', 'mean'), n=('du', 'size'),
                                  aod=('mean_u', 'mean')).reset_index()
py = py[py['n'] >= 20]
print('Plant-years with 20+ valid downwind/upwind day pairs:', len(py),
      ' plants:', py['fid'].nunique())
print(f'Mean down-minus-up AOD at operating plants: {py.du.mean():.4f} '
      f'(upwind AOD {py.aod.mean():.3f})')

# calibration from control plants: contrast vs SO2 emissions
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'so2mass'])
a['so2mass'] = pd.to_numeric(a['so2mass'], errors='coerce').fillna(0)
so2 = a.groupby(['facilityid', 'year'])['so2mass'].sum().rename('so2')
py = py.join(so2, on=['fid', 'yr']).dropna()
X = np.column_stack([np.ones(len(py)), py['so2'] / 1000])
b = np.linalg.lstsq(X, py['du'].values, rcond=None)[0]
print(f'Control cross-section: down-minus-up AOD = {b[0]:.4f} + '
      f'{b[1]:.5f} per 1,000 t SO2')

# placebo changes
D = {}
for f, g in py.groupby('fid'):
    g = g.set_index('yr')['du']
    for y in range(2005, 2019):
        if y - 1 in g.index and y + 1 in g.index:
            D[(f, y)] = g[y - 1] - g[y + 1]
E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
for thr in [2500, 5000, 10000]:
    ev = E[(E['so2_drop'] >= thr) & (E['event_year'].between(2005, 2018))]
    years = ev['event_year'].values
    rng = np.random.default_rng(20261109)
    plants = sorted({f for f, _ in D})
    dr = np.array([D[(f, int(y))] for _ in range(1000) for f in plants
                   for y in [rng.choice(years)] if (f, int(y)) in D])
    se = dr.std() / np.sqrt(len(ev))
    pred = (b[1] * ev['so2_drop'] / 1000).mean()
    z = pred / se if se > 0 else np.nan
    print(f'SO2 drop >= {thr:,} t (2005-2018): {len(ev)} events; placebo SD '
          f'{dr.std():.4f}; SE {se:.4f}; expected fall {pred:.4f} AOD = '
          f'{z:.1f} SE; power vs zero about '
          f'{100 * (Phi(z - 1.96) + Phi(-z - 1.96)):.0f}%')
pd.DataFrame({'d': dr}).to_csv(f'{DIR}/maiac_placebo_draws.csv', index=False)
