# Step: find European coal plants that stopped burning coal, and how much
# they still emitted at the end. EEA plant data 2016-2023. No outcomes.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd

d = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv',
                low_memory=False)
d['featureValue'] = pd.to_numeric(d['featureValue'], errors='coerce')
d['key'] = np.where(d['featureType'] == 'LCPCharacteristics',
                    d['unit'], d['featureType'])
w = d.pivot_table(index=['LCPInspireId', 'reportingYear'], columns='key',
                  values='featureValue', aggfunc='sum').reset_index()
info = (d.groupby('LCPInspireId')
         .agg(name=('installationPartName', 'first'),
              country=('countryName', 'first'),
              lat=('Latitude', 'first'), lon=('Longitude', 'first'))
         .reset_index())
for c in ['Coal', 'Lignite', 'Biomass', 'NaturalGas', 'LiquidFuels', 'Peat',
          'OtherGases', 'OtherSolidFuels', 'SO2', 'NOX', 'DUST', 'hours',
          'MW']:
    if c not in w:
        w[c] = 0.0
w = w.fillna(0)
w['solid'] = w['Coal'] + w['Lignite']
fuels = ['Coal', 'Lignite', 'Biomass', 'NaturalGas', 'LiquidFuels', 'Peat',
         'OtherGases', 'OtherSolidFuels']
w['energy'] = w[fuels].sum(axis=1)
yrs = sorted(w['reportingYear'].unique())
print('Years:', yrs)
print('Last year reported, by country:',
      d.groupby('countryName')['reportingYear'].max().to_dict())
LAST = {c: y for c, y in d.groupby('countryName')['reportingYear'].max()
        .items()}

# coal LCPs: coal + lignite at least half of energy input in some year
w['share'] = w['solid'] / w['energy'].replace(0, np.nan)
coal_ids = w.loc[(w['share'] >= 0.5) & (w['solid'] > 0), 'LCPInspireId']
cw = w[w['LCPInspireId'].isin(set(coal_ids))].merge(info, on='LCPInspireId')
print('Coal LCPs:', cw['LCPInspireId'].nunique())

# group units into sites: LCPs within 3 km of each other
L = (cw.groupby('LCPInspireId')[['lat', 'lon', 'name', 'country']]
       .first().reset_index())
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
site = -np.ones(len(L), int)
k = 0
for i in range(len(L)):
    if site[i] >= 0:
        continue
    dd = km(L.lat.values, L.lon.values, L.lat[i], L.lon[i])
    site[(dd <= 3) & (site < 0)] = k
    k += 1
L['site'] = site
cw = cw.merge(L[['LCPInspireId', 'site']], on='LCPInspireId')

# site-year totals (all LCPs at the site, any fuel)
allw = w.merge(info, on='LCPInspireId')
rows = []
for s, g in cw.groupby('site'):
    ids = g['LCPInspireId'].unique()
    lat, lon = g['lat'].mean(), g['lon'].mean()
    ctry = g['country'].iloc[0]
    sy = g.groupby('reportingYear')[['solid', 'SO2', 'NOX', 'DUST',
                                     'hours', 'MW']].sum()
    sy = sy.reindex(range(min(yrs), LAST[ctry] + 1)).fillna(0)
    burn = sy.index[sy['solid'] > 0]
    if not len(burn):
        continue
    last = int(burn.max())
    closed = last < LAST[ctry]
    peak_so2 = sy['SO2'].max()
    full = last - 1
    r = dict(site=s, name=g['name'].iloc[0], country=ctry,
             lat=round(lat, 4), lon=round(lon, 4), n_lcp=len(ids),
             mw_th=sy['MW'].max(), last_coal_year=last, closed=closed,
             so2_last=sy.at[last, 'SO2'], nox_last=sy.at[last, 'NOX'],
             so2_full=sy.at[full, 'SO2'] if full in sy.index else np.nan,
             nox_full=sy.at[full, 'NOX'] if full in sy.index else np.nan,
             dust_full=sy.at[full, 'DUST'] if full in sy.index else np.nan,
             so2_peak=peak_so2,
             so2_2016=sy.at[2016, 'SO2'] if 2016 in sy.index else np.nan,
             hours_full=sy.at[full, 'hours'] if full in sy.index else np.nan,
             tj_solid_full=sy.at[full, 'solid'] if full in sy.index
             else np.nan,
             tj_solid_peak=sy['solid'].max())
    if closed and last + 1 in sy.index:
        r['so2_after'] = sy.at[last + 1, 'SO2']
    rows.append(r)
S = pd.DataFrame(rows)
S['share_left'] = S['tj_solid_full'] / S['tj_solid_peak'].replace(0, np.nan)

# cross-reference our GEM target list
pl = pd.read_csv(f'{DIR}/plants.csv')
eu = pl[~pl['country'].isin(['United States', 'China'])]
def nearest(r):
    dd = km(eu.lat.values, eu.lon.values, r.lat, r.lon)
    j = int(np.argmin(dd))
    return pd.Series({'gem_match': eu['name'].iloc[j] if dd[j] <= 5
                      else '', 'gem_km': round(float(dd[j]), 1)})
S = pd.concat([S, S.apply(nearest, axis=1)], axis=1)
S.to_csv(f'{DIR}/eu_coal_sites.csv', index=False)

C = S[S['closed']].copy()
print(f'\nCoal sites: {len(S)}; stopped coal before their last reported '
      f'year: {len(C)}')
print('Closures by last coal year:',
      C['last_coal_year'].value_counts().sort_index().to_dict())
sat = C[C['last_coal_year'].between(2019, 2022)]
print(f'\nSATELLITE ERA (last coal year 2019-2022): {len(sat)} sites')
q = lambda s: s.quantile([.25, .5, .75]).round(0).tolist()
print('  SO2 in last full year, tonnes, quartiles:', q(sat['so2_full']))
print('  NOx in last full year:', q(sat['nox_full']))
print('  Coal use in last full year as share of 2016+ peak, median:',
      round(sat['share_left'].median(), 2))
for t in [1000, 2000, 5000, 10000]:
    print(f'  Sites with SO2 >= {t:,} t in last full year:',
          int((sat['so2_full'] >= t).sum()))
print('  By country:', sat['country'].value_counts().to_dict())

pd.set_option('display.width', 250)
cols = ['name', 'country', 'last_coal_year', 'mw_th', 'so2_peak',
        'so2_full', 'nox_full', 'so2_last', 'share_left', 'gem_match']
print('\nSatellite-era closures, by SO2 in last full year:')
print(sat.sort_values('so2_full', ascending=False)[cols]
      .round(2).head(45).to_string(index=False))
print('\nUS comparison: median SO2 in last 12 months before closure = 962 t')
