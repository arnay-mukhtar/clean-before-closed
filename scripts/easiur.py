# Robustness of the valuation results (Q1/Q2) to the damage model:
# EASIUR marginal social costs (Heo, Adams and Gao 2016; 2010 USD per metric
# ton, 150 m release, annual) in place of InMAP's.
# Version 2 (10 Nov 2026): the EASIUR site gives no direct file links, so
# this uses the plant-level EASIUR values published with eGRID 2016 plant
# coordinates in Deetjen and Azevedo's simple_dispatch repository
# (egrid_2016_plant_easiur.csv). Plants not in eGRID 2016 (mostly closed
# before 2016) take the value of the nearest eGRID plant within 36 km,
# one EASIUR grid cell.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, requests
import numpy as np, pandas as pd

OUT = f'{DIR}/easiur'
os.makedirs(OUT, exist_ok=True)
ST2T = 0.90718474
URL = ('https://raw.githubusercontent.com/tdeetjen/simple_dispatch/master/'
       'egrid_2016_plant_easiur.csv')
p = f'{OUT}/egrid_2016_plant_easiur.csv'
if not os.path.exists(p):
    r = requests.get(URL, timeout=120); r.raise_for_status()
    open(p, 'wb').write(r.content)
E = pd.read_csv(p)
E = E.dropna(subset=['LAT', 'LON', 'SO2 Annual 150m', 'NOX Annual 150m'])
E = E[(E['SO2 Annual 150m'] > 0) & (E['NOX Annual 150m'] > 0)]
print(f'EASIUR plant file: {len(E)} plants with values')

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
    .dropna(subset=['latitude', 'longitude'])
byoris = E.drop_duplicates('ORISPL').set_index('ORISPL')
rows = []
for f in fc.itertuples():
    if f.facilityid in byoris.index:
        e = byoris.loc[f.facilityid]; how, dist = 'exact', 0.0
    else:
        d = km(E['LAT'].values, E['LON'].values, f.latitude, f.longitude)
        j = int(np.argmin(d)); dist = float(d[j])
        if dist > 36:
            rows.append(dict(facilityid=f.facilityid, how='missing',
                             dist=dist)); continue
        e = E.iloc[j]; how = 'nearest'
    rows.append(dict(facilityid=f.facilityid, how=how, dist=dist,
                     e_so2=e['SO2 Annual 150m'], e_nox=e['NOX Annual 150m'],
                     cell=f.cell))
M = pd.DataFrame(rows).set_index('facilityid')

U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)
U = U.join(M, on='fid')
print('Exit units by match type:', U['how'].value_counts().to_dict(),
      f'(nearest-match median distance {U.loc[U.how == "nearest", "dist"].median():.1f} km)')
U = U[U['e_so2'].notna()]

mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
ok = U['cell'] >= 0
im = mv['MD_SO2_high'].values[U.loc[ok, 'cell'].astype(int)]
k = im > 0
print(f'Log correlation, EASIUR vs InMAP SO2 damages at exit plants: '
      f'{np.corrcoef(np.log(U.loc[ok, "e_so2"].values[k]), np.log(im[k]))[0, 1]:.2f}; '
      f'median ratio EASIUR/InMAP {np.median(U.loc[ok, "e_so2"].values[k] / im[k] / ST2T):.2f}')

ms, mn = U['e_so2'] * ST2T, U['e_nox'] * ST2T
f_d = U['so2_final'] * ms + U['nox_final'] * mn
print(f'\nEASIUR, 150 m release, {len(U)} unit exits:')
for base in ['t10', 'peak']:
    b_d = U[f'so2_{base}'] * ms + U[f'nox_{base}'] * mn
    kk = b_d > 0
    print(f'  baseline {base}: share of damage cut before the last full year '
          f'{100 * (1 - f_d[kk].sum() / b_d[kk].sum()):.0f}%; '
          f'baseline / last full year {b_d[kk].sum() / f_d[kk].sum():.2f}x')
cf = f'{DIR}/counterfactual_paths_v2.csv'
if os.path.exists(cf):
    C = U.merge(pd.read_csv(cf)[['uid', 'ratio_so2', 'ratio_nox']], on='uid')
    cs, cn = C['e_so2'] * ST2T, C['e_nox'] * ST2T
    fin = (C.so2_final * cs + C.nox_final * cn).sum()
    cfd = (C.so2_final * C.ratio_so2 * cs + C.nox_final * C.ratio_nox * cn).sum()
    t10 = (C.so2_t10 * cs + C.nox_t10 * cn).sum()
    print(f'  counterfactual ({len(C)} exits): last-full-year {fin / cfd:.2f}x, '
          f'ten-years-before {t10 / cfd:.2f}x')
print('\nInMAP (paper): 61% / 81%; 2.58x / 5.18x; counterfactual 1.71x / 4.01x')
