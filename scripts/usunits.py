# Step: every US coal unit that stopped burning coal since 2005, and how
# much of its emissions fall happened before that date. EPA annual unit
# data 1997-2025. Operation data only; no air-quality outcomes.
DIR = '/content/drive/MyDrive/coal_closures'
import os, time, requests
import numpy as np, pandas as pd

URL = ('https://api.epa.gov/easey/streaming-services/emissions/'
       'apportioned/annual')
YEARS = list(range(1997, 2026))
STATES = ('AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI '
          'MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX '
          'UT VT VA WA WV WI WY AK HI').split()
RAW = f'{DIR}/us_epa_annual_all.csv'

if os.path.exists(RAW):
    a = pd.read_csv(RAW, low_memory=False)
else:
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    parts = []
    for i, st in enumerate(STATES):
        for tries in range(6):
            r = requests.get(URL, params=dict(
                api_key=key, stateCode=st,
                year='|'.join(map(str, YEARS))), timeout=300)
            if r.status_code == 429:
                time.sleep(15 * (tries + 1)); continue
            if r.status_code >= 500:
                time.sleep(10); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json())
        if len(d):
            parts.append(d)
        print(f'{st}: {len(d)} rows', flush=True)
    a = pd.concat(parts)
    a.columns = [c.lower() for c in a.columns]
    a.to_csv(RAW, index=False)

for c in ['so2mass', 'noxmass', 'heatinput', 'grossload']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
a['coal'] = (a['primaryfuelinfo'].fillna('').str.contains('Coal')
             & (a['heatinput'] > 0))
print('Units with any coal year:', a.loc[a['coal'], 'uid'].nunique())

rows, traj = [], []
for uid, g in a[a['uid'].isin(set(a.loc[a['coal'], 'uid']))].groupby('uid'):
    g = g.groupby('year')[['so2mass', 'noxmass', 'heatinput', 'coal']] \
         .agg({'so2mass': 'sum', 'noxmass': 'sum', 'heatinput': 'sum',
               'coal': 'max'}).reindex(range(1990, 2026)).fillna(0)
    cy = g.index[g['coal'] > 0]
    if len(cy) < 5:
        continue
    last = int(cy.max())
    if last >= 2025 or last < 2005:
        continue                     # still burning coal, or too early
    t = g.copy(); t['k'] = t.index - last
    pre = t[t['k'] <= -1]
    peak = pre['so2mass'].max()
    rows.append(dict(
        uid=uid, last_coal_year=last,
        so2_peak=peak, so2_t10=g.at[last - 10, 'so2mass'],
        so2_0507=g.loc[2005:2007, 'so2mass'].mean(),
        so2_final=g.at[last - 1, 'so2mass'],
        so2_lastyr=g.at[last, 'so2mass'],
        so2_after=g.loc[last + 1:last + 2, 'so2mass'].mean()
        if last <= 2023 else np.nan,
        nox_peak=pre['noxmass'].max(), nox_final=g.at[last - 1, 'noxmass'],
        nox_t10=g.at[last - 10, 'noxmass'],
        heat_t10=g.at[last - 10, 'heatinput'],
        heat_final=g.at[last - 1, 'heatinput'],
        heat_peak=pre['heatinput'].max(),
        converted=bool(g.loc[last + 1:, 'heatinput'].sum() > 0)))
    t['uid'] = uid
    traj.append(t[(t['k'] >= -15) & (t['k'] <= 2)].reset_index())
U = pd.DataFrame(rows)
T = pd.concat(traj)
U.to_csv(f'{DIR}/us_unit_exits.csv', index=False)
T.to_csv(f'{DIR}/us_unit_exit_trajectories.csv', index=False)

print(f'\nCoal unit exits 2005-2024: {len(U)} '
      f'(converted to other fuel: {int(U.converted.sum())})')
print('By last coal year:',
      U['last_coal_year'].value_counts().sort_index().to_dict())

def share_before(base, final):
    b, f = U[base], U[final]
    ok = b > 0
    return 1 - f[ok].sum() / b[ok].sum(), (1 - f[ok] / b[ok]).median()

print('\nSHARE OF THE EMISSIONS FALL THAT HAPPENED BEFORE THE LAST FULL '
      'YEAR (tonnage-weighted, unit median)')
for lab, b, f in [('SO2 vs pre-exit peak', 'so2_peak', 'so2_final'),
                  ('SO2 vs 10 years before', 'so2_t10', 'so2_final'),
                  ('SO2 vs 2005-07 average', 'so2_0507', 'so2_final'),
                  ('NOx vs pre-exit peak', 'nox_peak', 'nox_final'),
                  ('NOx vs 10 years before', 'nox_t10', 'nox_final'),
                  ('Heat input vs 10 years before', 'heat_t10',
                   'heat_final')]:
    w, m = share_before(b, f)
    print(f'  {lab}: {100 * w:.0f}% weighted, {100 * m:.0f}% median')

# how much of the SO2 fall came from running less vs emitting less per unit
ok = ((U['heat_t10'] > 0) & (U['so2_t10'] > 0) & (U['heat_final'] > 0)
      & (U['so2_final'] > 0)
      & (U['so2_final'] < 0.9 * U['so2_t10']))
r0 = U.loc[ok, 'so2_t10'] / U.loc[ok, 'heat_t10']
r1 = U.loc[ok, 'so2_final'] / U.loc[ok, 'heat_final']
h = np.log(U.loc[ok, 'heat_final'] / U.loc[ok, 'heat_t10'])
r = np.log(r1 / r0)
w = U.loc[ok, 'so2_t10']
print('\nSO2 fall from t-10 to the last full year, decomposed (weighted '
      'by t-10 SO2):')
print(f'  running less (heat input): {100 * np.average(h / (h + r), weights=w):.0f}%'
      f'   cleaner per unit of fuel: {100 * np.average(r / (h + r), weights=w):.0f}%')

# event-time profile: total SO2, NOx and heat input across exits that have
# the full window, relative to 10 years before
full = T.groupby('uid')['k'].min() <= -10
Tf = T[T['uid'].isin(full[full].index)]
prof = Tf.groupby('k')[['so2mass', 'noxmass', 'heatinput']].sum()
prof = prof / prof.loc[-10]
print(f'\nEvent-time totals relative to t-10 ({full.sum()} units with the '
      'full window):')
print(prof.loc[-10:2].round(2).T.to_string())

print('\nLast-full-year SO2 per exiting unit, tonnes, quartiles:',
      U['so2_final'].quantile([.25, .5, .75]).round(0).tolist())
print('Exits with last-full-year SO2 >= 5,000 t:',
      int((U['so2_final'] >= 5000).sum()), ' >= 10,000 t:',
      int((U['so2_final'] >= 10000).sum()))
