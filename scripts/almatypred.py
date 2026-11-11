# H4 ADDENDUM: modelled prediction for Almaty CHP-2 (coal to gas), to be
# filed on OSF before any post-commissioning data is viewed (registration
# osf.io/v7s6e, section 5 Q4). Uses ONLY pre-period data:
#   - TROPOMI tropospheric NO2, January-March 2025 and 2026 (the registered
#     pre-period), near disk 0-10 km and far ring 25-50 km around CHP-2;
#   - ERA5-Land hourly 10 m wind at the plant at 08 UTC (TROPOMI overpass,
#     about 13:30 local solar time), January-March 2019-2026.
# HARD GUARD: nothing dated on or after 1 April 2026 is requested.
#
# Model (steady-state point source, isotropic over wind directions): plant
# NO2 mass within distance r of the stack is E * tau * (1 - exp(-r/(u*tau))),
# so its mean column over the 0-10 km disk is M(10)/(pi 10^2) and over the
# 25-50 km ring (M(50) - M(25))/(pi (50^2 - 25^2)). Daily columns use daily
# wind; the season mean averages over days. The predicted change in the
# registered contrast (log near - log far, pre minus post, positive = fall)
# removes the emission cut from the observed pre-period columns.
# Inputs and ranges (Monte Carlo, 20,000 draws):
#   coal NO2 emissions 7.7-8.7 kt/yr (CHP-2, 2012 and 2019, KazNIPIEnergoprom
#     /ALES figures); gas NO2 3.07 kt/yr x U(0.7, 1.3) (project gas option);
#   January-March emission rate = annual rate x U(1.2, 1.6) (heating load);
#   effective plume height U(150, 400) m, wind scaled from 10 m by power law,
#     exponent U(0.14, 0.35) (winter, stable);
#   NOx lifetime U(6, 24) h; NO2 share of plume NOx U(0.5, 0.8) (near-stack
#   plumes are NO-rich); TROPOMI sensitivity to the plume layer U(0.5, 1.0).
# Also reports the prediction if only part of the coal NOx is removed.
import numpy as np, pandas as pd, ee
try:
    ee.Initialize(project='coal-closures-air')
except Exception:
    ee.Authenticate(); ee.Initialize(project='coal-closures-air')

LAT, LON = 43.29292, 76.798519          # Almaty-2 (GEM, exact)
CUTOFF = '2026-04-01'                    # nothing on or after this date
pt = ee.Geometry.Point([LON, LAT])
near = pt.buffer(10000)
far = pt.buffer(50000).difference(pt.buffer(25000))

no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2') \
    .select('tropospheric_NO2_column_number_density')
def season(y):
    ic = no2.filterDate(f'{y}-01-01', f'{y}-04-01')
    def per(img):
        r = img.reduceRegions(ee.FeatureCollection([
            ee.Feature(near, {'zone': 'near'}),
            ee.Feature(far, {'zone': 'far'})]),
            ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            1113)
        return r.map(lambda f: f.set('date', img.date().format('YYYY-MM-dd'))
                     .setGeometry(None))
    return ee.FeatureCollection(ic.map(per)).flatten()
rows = []
for y in [2025, 2026]:
    assert f'{y}-04-01' <= CUTOFF
    fc = season(y).filter(ee.Filter.notNull(['mean'])).getInfo()
    rows += [dict(year=y, **f['properties']) for f in fc['features']]
d = pd.DataFrame(rows)
d = d[d['count'] >= 20]
w = d.pivot_table(index=['year', 'date'], columns='zone', values='mean') \
     .dropna()
print(f'TROPOMI valid days (both zones): 2025 {len(w.loc[2025])}, 2026 '
      f'{len(w.loc[2026])}')
S = w.groupby('year').mean()
S['contrast'] = np.log(S['near']) - np.log(S['far'])
print('Pre-period season means (mol/m2):')
print(S.round(7).to_string())
N0, F0 = S['near'].mean(), S['far'].mean()

# winds at 08 UTC, Jan-Mar 2019-2026
era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])
ws = []
for y in range(2019, 2027):
    ic = era.filterDate(f'{y}-01-01', f'{y}-04-01') \
            .filter(ee.Filter.calendarRange(8, 8, 'hour'))
    fc = ic.map(lambda im: ee.Feature(None, im.reduceRegion(
        ee.Reducer.first(), pt, 11132)).set('t', im.date().millis())) \
        .getInfo()
    ws += [f['properties'] for f in fc['features']]
u10 = np.array([np.hypot(p['u_component_of_wind_10m'],
                         p['v_component_of_wind_10m']) for p in ws
                if p.get('u_component_of_wind_10m') is not None])
print(f'\nERA5-Land 10 m wind at 08 UTC, Jan-Mar 2019-2026: {len(u10)} '
      f'hours; median {np.median(u10):.2f} m/s')

rng = np.random.default_rng(20261109)
K = 20000
KG_PER_MOL = 0.046
SEC_YR = 365.25 * 86400
Ecoal = rng.uniform(7.7e6, 8.7e6, K)                 # kg NO2 / yr
Egas = 3.07e6 * rng.uniform(0.7, 1.3, K)
wf = rng.uniform(1.2, 1.6, K)
z = rng.uniform(150, 400, K)
al = rng.uniform(0.14, 0.35, K)
tau = rng.uniform(6, 24, K) * 3600
k = rng.uniform(0.5, 1.0, K) * rng.uniform(0.5, 0.8, K)
ud = np.clip(u10, 0.5, None)
def cols(E, i):
    """Mean plant columns (mol/m2) near and far for draw i, emission E."""
    u = ud * (z[i] / 10) ** al[i]
    q = E * wf[i] / SEC_YR                            # kg/s
    M = lambda r: q * tau[i] * (1 - np.exp(-r / (u * tau[i])))
    cn = (M(10e3) / (np.pi * 10e3 ** 2)).mean()
    cf = ((M(50e3) - M(25e3)) / (np.pi * (50e3 ** 2 - 25e3 ** 2))).mean()
    return k[i] * cn / KG_PER_MOL, k[i] * cf / KG_PER_MOL
pred, share = np.empty(K), np.empty(K)
frac_tab = {f: np.empty(K) for f in [0.25, 0.5, 0.75]}
for i in range(K):
    cn_c, cf_c = cols(Ecoal[i], i)
    cn_g, cf_g = cols(Egas[i], i)
    share[i] = cn_c / N0
    dn, df = cn_c - cn_g, cf_c - cf_g
    pred[i] = (np.log(N0) - np.log(F0)) - (np.log(max(N0 - dn, 1e-9))
                                           - np.log(max(F0 - df, 1e-9)))
    for f in frac_tab:
        frac_tab[f][i] = (np.log(N0) - np.log(F0)) - (
            np.log(max(N0 - f * dn, 1e-9)) - np.log(max(F0 - f * df, 1e-9)))
q = lambda x: np.percentile(x, [5, 50, 95])
print('\nPREDICTION (pre minus post, log points; positive = near-field NO2 '
      'falls relative to 25-50 km):')
print('  full switch to gas: median %.3f, 90%% interval [%.3f, %.3f]'
      % tuple(q(pred)[[1, 0, 2]]))
for f, v in frac_tab.items():
    print(f'  if only {int(f * 100)}%% of the coal-to-gas NOx cut happens: '
          'median %.3f, 90%% interval [%.3f, %.3f]' % tuple(q(v)[[1, 0, 2]]))
print('  modelled share of the pre-period near-disk column from CHP-2: '
      'median %.2f, 90%% interval [%.2f, %.2f]' % tuple(q(share)[[1, 0, 2]]))
pd.DataFrame({'pred': pred, 'share': share}).to_csv(
    '/content/drive/MyDrive/coal_closures/almaty_prediction_draws.csv',
    index=False)
print('\nSaved almaty_prediction_draws.csv. Send this output; it goes into '
      'the addendum.')
