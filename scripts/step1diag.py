# Diagnose the primary PM2.5 gaps and outliers for the 47 primary events.
# Emissions data only.
DIR = '/content/drive/MyDrive/coal_closures'
import zipfile
import numpy as np, pandas as pd

zz = zipfile.ZipFile(f'{DIR}/step1/echo_air_emissions.zip')
name = [n for n in zz.namelist() if n.lower().endswith('.csv')][0]
camd, eis = [], []
for ch in pd.read_csv(zz.open(name), chunksize=2_000_000, dtype=str,
                      encoding='latin-1'):
    ch.columns = [c.upper() for c in ch.columns]
    camd.append(ch[ch['PGM_SYS_ACRNM'] == 'CAMDBS']
                [['REGISTRY_ID', 'PGM_SYS_ID']].drop_duplicates())
    eis.append(ch[(ch['PGM_SYS_ACRNM'] == 'EIS')
                  & ch['POLLUTANT_NAME'].isin(
                      ['Primary PM2.5 (filterables and condensibles)',
                       'Primary PM2.5, filterable portion only'])])
camd = pd.concat(camd).drop_duplicates()
camd['oris'] = pd.to_numeric(camd['PGM_SYS_ID'], errors='coerce')
eis = pd.concat(eis)
print('EIS PM rows:', len(eis), ' duplicated full rows:',
      int(eis.duplicated().sum()))
print('Columns:', list(eis.columns))
print('NEI_TYPE values:', eis['NEI_TYPE'].value_counts().to_dict())

P = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
pr = P[(P['n_near_bg'] > 0) & (P['so2_drop'] >= 5000)]
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
H = a.groupby(['facilityid', 'year'])['heatinput'].sum()

pd.set_option('display.width', 250, 'display.max_colwidth', 40)
rows = []
for r in pr.itertuples():
    regs = camd.loc[camd['oris'] == r.fid, 'REGISTRY_ID'].unique()
    e = eis[eis['REGISTRY_ID'].isin(regs)]
    tot = e[e['POLLUTANT_NAME'].str.contains('condens')]
    yrs = sorted(tot['REPORTING_YEAR'].unique())
    eisids = sorted(e['PGM_SYS_ID'].unique())
    info = dict(fid=r.fid, name=r.name[:28], event=r.event_year,
                n_registry=len(regs), n_eis_ids=len(eisids),
                nei_years=','.join(yrs), source=r.pm25_source)
    for y in yrs:
        v = (pd.to_numeric(tot.loc[tot['REPORTING_YEAR'] == y,
                                   'ANNUAL_EMISSION'], errors='coerce'))
        h = H.get((r.fid, int(y)), 0)
        info[f'lb_per_mmbtu_{y}'] = (v.sum() / h) if h > 0 else np.nan
        info[f'rows_{y}'] = len(v)
    rows.append(info)
D = pd.DataFrame(rows)
print(D[['fid', 'name', 'event', 'n_registry', 'n_eis_ids', 'nei_years',
         'source']].to_string(index=False))
rate_cols = sorted(c for c in D.columns if c.startswith('lb_per'))
print('\nPM2.5 rate, lb per mmBtu (facility total / facility heat input):')
print(D[['name', 'event'] + rate_cols].round(3).to_string(index=False))
row_cols = sorted(c for c in D.columns if c.startswith('rows_'))
print('\nRows per facility-year (more than 1 suggests duplicates):')
print(D[['name'] + row_cols].to_string(index=False))
