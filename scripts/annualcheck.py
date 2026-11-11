# Check: is the EPA annual unit file complete? Gorgas (plant 8) appears to
# stop in 2005, but it burned coal until 2019. Re-pull year by year
# (national, one request per year) and compare. Emissions data only.
DIR = '/content/drive/MyDrive/coal_closures'
import os, time, requests
import numpy as np, pandas as pd

old = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False)
old['so2mass'] = pd.to_numeric(old['so2mass'], errors='coerce')
print('OLD file: years for plant 8 (Gorgas):',
      sorted(old.loc[old['facilityid'] == 8, 'year'].unique()))
print('OLD file: rows by year:', old.groupby('year').size().to_dict())

NEW = f'{DIR}/us_epa_annual_all_v2.csv'
if not os.path.exists(NEW):
    from google.colab import userdata
    key = userdata.get('EPA_KEY')
    URL = ('https://api.epa.gov/easey/streaming-services/emissions/'
           'apportioned/annual')
    parts = []
    for y in range(1997, 2026):
        for tries in range(8):
            try:
                r = requests.get(URL, params=dict(api_key=key, year=str(y)),
                                 timeout=600)
            except requests.RequestException:
                time.sleep(20); continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(20 * (tries + 1)); continue
            r.raise_for_status(); break
        d = pd.DataFrame(r.json())
        d.columns = [c.lower() for c in d.columns]
        parts.append(d)
        print(y, len(d), 'rows', flush=True)
    new = pd.concat(parts)
    new.to_csv(NEW, index=False)
new = pd.read_csv(NEW, low_memory=False)
new['so2mass'] = pd.to_numeric(new['so2mass'], errors='coerce')

print('\nNEW file: years for plant 8 (Gorgas):',
      sorted(new.loc[new['facilityid'] == 8, 'year'].unique()))
cmp = pd.DataFrame({
    'rows_old': old.groupby('year').size(),
    'rows_new': new.groupby('year').size(),
    'so2_old_Mt': old.groupby('year')['so2mass'].sum() / 1e6,
    'so2_new_Mt': new.groupby('year')['so2mass'].sum() / 1e6})
print('\nRows and national SO2 (million short tons) by year:')
print(cmp.round(2).to_string())
print('\nReference points (EPA power sector SO2): about 10.2 Mt in 2005, '
      '5.1 in 2010, 1.5 in 2016')
fo = old.groupby('facilityid')['year'].nunique()
fn = new.groupby('facilityid')['year'].nunique()
j = pd.concat([fo, fn], axis=1, keys=['old', 'new']).fillna(0)
print('\nFacilities with more years in the new file:',
      int((j['new'] > j['old']).sum()), 'of', len(j))
print('State totals of rows, old vs new (top differences):')
so = old.groupby('statecode').size(); sn = new.groupby('statecode').size()
dd = pd.concat([so, sn], axis=1, keys=['old', 'new']).fillna(0)
dd['diff'] = dd['new'] - dd['old']
print(dd.sort_values('diff', ascending=False).head(12).to_string())
