# Step: the "fixed base year" valuation used by advocacy tallies (per-plant
# burdens from 2010, credited to plants that retired later) against what
# each retirement actually removed (last full year). Emissions + InMAP only.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd

ST2T = 0.90718474
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False)
for c in ['so2mass', 'noxmass']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
e10 = a[a['year'] == 2010].groupby('uid')[['so2mass', 'noxmass']].sum()

U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
U = U[U['last_coal_year'].between(2011, 2024)].copy()
U = U.join(e10.rename(columns={'so2mass': 'so2_2010',
                               'noxmass': 'nox_2010'}), on='uid')
U[['so2_2010', 'nox_2010']] = U[['so2_2010', 'nox_2010']].fillna(0)
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
G = np.load(f'{DIR}/isrm_grid.npz')
if not np.allclose(mv['Location_W'].values, G['W']):
    raise SystemExit('Grid order differs; tell me.')

print(f'Units retiring 2011-2024: {len(U)}')
for h in ['high', 'low']:
    c = U['cell'].values
    f = lambda so2, nox, kind: ((U[so2] * mv[f'{kind}_SO2_{h}'].values[c]
                                 + U[nox] * mv[f'{kind}_NOx_{h}'].values[c])
                                * ST2T)
    d10, dfin = f('so2_2010', 'nox_2010', 'MD'), f('so2_final', 'nox_final',
                                                   'MD')
    m10, mfin = f('so2_2010', 'nox_2010', 'MM'), f('so2_final', 'nox_final',
                                                   'MM')
    print(f'\n{h} stack:')
    print(f'  Credited with 2010 emissions: {m10.sum():,.0f} deaths/yr, '
          f'${d10.sum() / 1e9:.0f}bn/yr')
    print(f'  Removed by the retirement (last full year): {mfin.sum():,.0f} '
          f'deaths/yr, ${dfin.sum() / 1e9:.0f}bn/yr')
    print(f'  Ratio: {m10.sum() / mfin.sum():.1f}x')
    U['r'] = m10 / mfin.replace(0, np.nan)
    U['m10'], U['mfin'] = m10, mfin
    g = U.groupby(pd.cut(U['last_coal_year'], [2010, 2014, 2016, 2019,
                                               2024]))
    print('  By retirement period: deaths/yr credited (2010) vs removed, '
          'ratio')
    for k, x in g:
        print(f'    {k}: {x.m10.sum():,.0f} vs {x.mfin.sum():,.0f} '
              f'({x.m10.sum() / max(x.mfin.sum(), 1e-9):.1f}x), '
              f'{len(x)} units')
