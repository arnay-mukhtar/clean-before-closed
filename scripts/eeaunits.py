# Step: (1) the wind-down fact for large European closures; (2) unit-level
# coal exits, including units that close at plants that keep running, and
# the drop in the whole plant's emissions around each exit. No outcomes.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd

d = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv',
                low_memory=False)
d['featureValue'] = pd.to_numeric(d['featureValue'], errors='coerce')
d['key'] = np.where(d['featureType'] == 'LCPCharacteristics',
                    d['unit'], d['featureType'])
w = d.pivot_table(index=['LCPInspireId', 'reportingYear'], columns='key',
                  values='featureValue', aggfunc='sum').reset_index()
w = w.fillna(0)
for c in ['Coal', 'Lignite', 'SO2', 'NOX', 'DUST', 'MW', 'hours']:
    if c not in w:
        w[c] = 0.0
fuels = [c for c in ['Coal', 'Lignite', 'Biomass', 'NaturalGas',
                     'LiquidFuels', 'Peat', 'OtherGases', 'OtherSolidFuels']
         if c in w]
w['solid'] = w['Coal'] + w['Lignite']
w['energy'] = w[fuels].sum(axis=1)
info = (d.groupby('LCPInspireId')
         .agg(name=('installationPartName', 'first'),
              country=('countryName', 'first'),
              lat=('Latitude', 'first'), lon=('Longitude', 'first'))
         .reset_index())
LAST = d.groupby('countryName')['reportingYear'].max().to_dict()
FIRST = int(d['reportingYear'].min())

w['share'] = w['solid'] / w['energy'].replace(0, np.nan)
coal_ids = set(w.loc[(w['share'] >= 0.5) & (w['solid'] > 0),
                     'LCPInspireId'])

# sites = all LCPs (any fuel) within 3 km of a coal LCP cluster
I = info.copy()
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
C = I[I['LCPInspireId'].isin(coal_ids)].reset_index(drop=True)
site = -np.ones(len(C), int); k = 0
for i in range(len(C)):
    if site[i] >= 0:
        continue
    dd = km(C.lat.values, C.lon.values, C.lat[i], C.lon[i])
    site[(dd <= 3) & (site < 0)] = k; k += 1
C['site'] = site
cen = C.groupby('site')[['lat', 'lon']].mean()
# attach non-coal LCPs at the same site (they keep emitting after a switch)
I['site'] = -1
for s, r in cen.iterrows():
    dd = km(I.lat.values, I.lon.values, r.lat, r.lon)
    I.loc[(dd <= 3) & (I['site'] < 0), 'site'] = s
W = w.merge(I[['LCPInspireId', 'site', 'name', 'country']],
            on='LCPInspireId')
W = W[W['site'] >= 0]
SY = W.groupby(['site', 'reportingYear'])[['SO2', 'NOX', 'DUST', 'solid',
                                           'MW']].sum()

ev = []
for lcp, g in W[W['LCPInspireId'].isin(coal_ids)].groupby('LCPInspireId'):
    g = g.set_index('reportingYear').sort_index()
    ctry = g['country'].iloc[0]
    burn = g.index[g['solid'] > 0]
    if not len(burn):
        continue
    last = int(burn.max())
    if last >= LAST[ctry]:
        continue                       # still burning coal
    s = int(g['site'].iloc[0])
    sy = SY.loc[s].reindex(range(FIRST, LAST[ctry] + 1)).fillna(0)
    before, after = last - 1, last + 1
    if before < FIRST:
        continue
    unit = g.reindex(range(FIRST, LAST[ctry] + 1)).fillna(0)
    site_coal_after = sy.at[after, 'solid']
    ev.append(dict(
        lcp=lcp, name=g['name'].iloc[0], country=ctry, site=s,
        last_coal_year=last, unit_mw_th=unit['MW'].max(),
        unit_so2_before=unit.at[before, 'SO2'],
        unit_nox_before=unit.at[before, 'NOX'],
        unit_so2_peak=unit['SO2'].max(),
        unit_coal_share_left=(unit.at[before, 'solid']
                              / max(unit['solid'].max(), 1e-9)),
        site_so2_before=sy.at[before, 'SO2'],
        site_so2_after=sy.at[after, 'SO2'],
        site_nox_before=sy.at[before, 'NOX'],
        site_nox_after=sy.at[after, 'NOX'],
        full_site_exit=site_coal_after == 0))
E = pd.DataFrame(ev)
E['site_so2_drop'] = E['site_so2_before'] - E['site_so2_after']
E['site_nox_drop'] = E['site_nox_before'] - E['site_nox_after']
E.to_csv(f'{DIR}/eu_unit_exits.csv', index=False)

# ---------- (1) wind-down fact, large units ----------
big = E[E['unit_mw_th'] >= 1000]
print(f'Coal unit exits (any year): {len(E)}; units >= 1,000 MWth: '
      f'{len(big)}')
print('Large units: coal use in last full year as share of 2016+ peak, '
      'quartiles:', big['unit_coal_share_left'].quantile([.25, .5, .75])
      .round(2).tolist())
print('Large units: SO2 in last full year as share of 2016+ peak, median:',
      round((big['unit_so2_before'] / big['unit_so2_peak'].replace(0, np.nan))
            .median(), 2))

# ---------- (2) dose: site emission drop around each exit ----------
sat = (E[E['last_coal_year'].between(2019, 2022)]
       .sort_values('unit_mw_th', ascending=False)
       .drop_duplicates(['site', 'last_coal_year']))   # one event per site-year
print(f'\nSite-year exit events, last coal year 2019-2022: {len(sat)} '
      f'(full site exits {int(sat.full_site_exit.sum())}, partial '
      f'{int((~sat.full_site_exit).sum())})')
for col, lab in [('site_so2_drop', 'SO2'), ('site_nox_drop', 'NOx')]:
    for t in [1000, 2000, 5000, 10000]:
        print(f'  Exits where site {lab} fell by >= {t:,} t '
              '(year before vs year after):', int((sat[col] >= t).sum()))
print('  By country (NOx drop >= 1,000 t):',
      sat[sat['site_nox_drop'] >= 1000]['country'].value_counts().to_dict())

pd.set_option('display.width', 250)
cols = ['name', 'country', 'last_coal_year', 'unit_mw_th', 'full_site_exit',
        'unit_coal_share_left', 'site_so2_before', 'site_so2_after',
        'site_nox_before', 'site_nox_after']
print('\nLargest NOx drops around unit exits, 2019-2022:')
print(sat.sort_values('site_nox_drop', ascending=False)[cols]
      .round(2).head(40).to_string(index=False))
