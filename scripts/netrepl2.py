# POST-REGISTRATION, exploratory: how large could replacement emissions be?
# netrepl.py finds that 36-62% of the heat input lost at exits came back at
# other units in the same state or census division (lead placebos near
# zero), but the emissions regressions fail their placebos because exits
# coincided with region-wide cuts. This script bounds replacement emissions
# directly: replaced fuel (theta x the exit's last-full-year heat input) is
# emitted at the average rate of the other fossil units in the same region
# in the year after exit, valued with InMAP marginal mortality, and compared
# with the deaths removed under the last-full-year rule. theta = 0.38
# (state), 0.62 (division), 1 (all lost output replaced by fossil units);
# a dirtier bound uses the average rate of other COAL units only.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd
ST2T = 0.90718474

a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'statecode',
                         'so2mass', 'noxmass', 'heatinput',
                         'primaryfuelinfo'])
for c in ['so2mass', 'noxmass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
a['coalunit'] = a['primaryfuelinfo'].fillna('').str.contains('Coal')
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
fc = fc[fc['cell'] >= 0].copy()
fc['ms'] = mv['MM_SO2_high'].values[fc['cell'].astype(int)] * ST2T
fc['mn'] = mv['MM_NOx_high'].values[fc['cell'].astype(int)] * ST2T
fst = a.drop_duplicates('facilityid').set_index('facilityid')['statecode']
fc['state'] = fc['facilityid'].map(fst)
own = fc.set_index('facilityid')
a['ms'] = a['facilityid'].map(own['ms']).fillna(
    a['statecode'].map(fc.groupby('state')['ms'].mean())).fillna(fc['ms'].mean())
a['mn'] = a['facilityid'].map(own['mn']).fillna(
    a['statecode'].map(fc.groupby('state')['mn'].mean())).fillna(fc['mn'].mean())
a['deaths'] = a['so2mass'] * a['ms'] + a['noxmass'] * a['mn']
DIV = {'New England': 'CT ME MA NH RI VT', 'Mid-Atlantic': 'NJ NY PA',
       'East North Central': 'IL IN MI OH WI',
       'West North Central': 'IA KS MN MO NE ND SD',
       'South Atlantic': 'DE DC FL GA MD NC SC VA WV',
       'East South Central': 'AL KY MS TN',
       'West South Central': 'AR LA OK TX',
       'Mountain': 'AZ CO ID MT NV NM UT WY', 'Pacific': 'CA OR WA'}
s2d = {s: d for d, ss in DIV.items() for s in ss.split()}
a['state'] = a['statecode']
a['division'] = a['statecode'].map(s2d)

U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
U = U[U['cell'] >= 0].copy()
U['state'] = U['uid'].map(a.drop_duplicates('uid').set_index('uid')['statecode'])
U['division'] = U['state'].map(s2d)
U['ms'] = mv['MM_SO2_high'].values[U['cell'].astype(int)] * ST2T
U['mn'] = mv['MM_NOx_high'].values[U['cell'].astype(int)] * ST2T
U['removed'] = U['so2_final'] * U['ms'] + U['nox_final'] * U['mn']
exiting = set(U['uid'])

for region in ['state', 'division']:
    oth = a[~a['uid'].isin(exiting) & a[region].notna()]
    allr = oth.groupby([region, 'year'])[['deaths', 'heatinput']].sum()
    coal = oth[oth['coalunit']].groupby([region, 'year'])[['deaths',
                                                           'heatinput']].sum()
    r_all = (allr['deaths'] / allr['heatinput']).rename('r_all')
    r_coal = (coal['deaths'] / coal['heatinput']).rename('r_coal')
    d = U.copy()
    key = list(zip(d[region], d['last_coal_year'] + 1))
    d['r_all'] = [r_all.get(k, np.nan) for k in key]
    d['r_coal'] = [r_coal.get(k, np.nan) for k in key]
    d = d.dropna(subset=['r_all', 'r_coal'])
    rem = d['removed'].sum()
    own_rate = rem / d['heat_final'].sum()
    print(f'\n{region}: {len(d)} exits. Exiting units emitted '
          f'{own_rate * 1e6:.2f} deaths per million mmBtu in their last full '
          f'year; other fossil units in the region {np.average(d.r_all, weights=d.heat_final) * 1e6:.2f}, '
          f'other coal units {np.average(d.r_coal, weights=d.heat_final) * 1e6:.2f}.')
    for th in [0.38, 0.62, 1.0]:
        ra = th * (d['heat_final'] * d['r_all']).sum() / rem
        rc = th * (d['heat_final'] * d['r_coal']).sum() / rem
        print(f'  theta {th:.2f}: replacement deaths = {100 * ra:.0f}% of '
              f'deaths removed at the average fossil rate, {100 * rc:.0f}% at '
              f'the average coal rate')
print('\nNet benefit = (1 - replacement share) x gross. Against the counter'
      'factual, the overstatement factors rise by 1/(1 - share) if '
      'replacement scales with the output the unit would have produced.')
