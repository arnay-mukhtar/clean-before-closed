# POST-REGISTRATION, exploratory (descriptive Q1 extension; no outcome data).
#  (A) Does the wind-down differ by how a plant is paid and dispatched?
#      Regulated utility plants (cost-of-service) vs independent producers,
#      and plants inside organised wholesale markets (RTO/ISO) vs outside.
#      This is the US analogue of the Q5 question for contracted systems.
#  (B) Q1 shares when exits are dated by the official EIA retirement year
#      instead of the last year of measured coal burning.
# Downloads EIA-860 (2009-2024) from eia.gov, keeps the plant and generator
# tables in coal_closures/eia860_plant_panel.csv and eia860_retired.csv.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, re, zipfile, requests
import numpy as np, pandas as pd

OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261109)
NB = 4000
COAL = {'BIT', 'SUB', 'LIG', 'RC', 'WC', 'ANT', 'SC', 'SGC'}
H = {'User-Agent': 'Mozilla/5.0 (research download)'}

def find_col(cols, *keys, avoid=()):
    for c in cols:
        lc = str(c).lower()
        if all(k in lc for k in keys) and not any(a in lc for a in avoid):
            return c
    return None

def read_sheet(raw):
    """EIA sheets have 0-2 title rows above the header."""
    for i in range(6):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains('plant code|plant id|plant_code').any():
            d = raw.iloc[i + 1:].copy(); d.columns = raw.iloc[i].astype(str)
            return d
    return None

p_pl, p_rt = f'{DIR}/eia860_plant_panel.csv', f'{DIR}/eia860_retired.csv'
if not (os.path.exists(p_pl) and os.path.exists(p_rt)):
    plants, retired = [], []
    for y in range(2009, 2025):
        content = None
        for base in ['https://www.eia.gov/electricity/data/eia860/xls/',
                     'https://www.eia.gov/electricity/data/eia860/archive/'
                     'xls/']:
            try:
                r = requests.get(f'{base}eia860{y}.zip', headers=H,
                                 timeout=300)
                if r.ok and r.content[:2] == b'PK':
                    content = r.content; break
            except Exception:
                pass
        if content is None:
            print(y, 'EIA-860 not found'); continue
        zz = zipfile.ZipFile(io.BytesIO(content))
        names = zz.namelist()
        pf = [n for n in names if re.search(r'plant', n, re.I)
              and n.lower().endswith(('.xlsx', '.xls'))
              and not re.search(r'owner|wind|solar|multi', n, re.I)]
        gf = [n for n in names if re.search(r'generator', n, re.I)
              and n.lower().endswith(('.xlsx', '.xls'))
              and not re.search(r'wind|solar|multi|storage', n, re.I)]
        for n in pf[:1]:
            raw = pd.read_excel(io.BytesIO(zz.read(n)), header=None)
            d = read_sheet(raw)
            if d is None:
                continue
            c = list(d.columns)
            o = pd.DataFrame({
                'year': y,
                'oris': pd.to_numeric(d[find_col(c, 'plant', 'code')
                                        or find_col(c, 'plant', 'id')],
                                      errors='coerce'),
                'sector': d[find_col(c, 'sector', 'name')].astype(str)
                if find_col(c, 'sector', 'name') else np.nan,
                'regstatus': d[find_col(c, 'regulatory')].astype(str)
                if find_col(c, 'regulatory') else np.nan,
                'ba': d[find_col(c, 'balancing', 'code')
                        or find_col(c, 'balancing')].astype(str)
                if (find_col(c, 'balancing', 'code')
                    or find_col(c, 'balancing')) else np.nan,
                'nerc': d[find_col(c, 'nerc')].astype(str)
                if find_col(c, 'nerc') else np.nan})
            plants.append(o.dropna(subset=['oris']))
        if y == 2024 or y == max(range(2009, 2025)):
            for n in gf[:1]:
                book = pd.read_excel(io.BytesIO(zz.read(n)), sheet_name=None,
                                     header=None)
                for sh, raw in book.items():
                    if not re.search(r'retired', sh, re.I):
                        continue
                    d = read_sheet(raw)
                    c = list(d.columns)
                    retired.append(pd.DataFrame({
                        'oris': pd.to_numeric(d[find_col(c, 'plant', 'code')],
                                              errors='coerce'),
                        'gen': d[find_col(c, 'generator', 'id')].astype(str),
                        'fuel': d[find_col(c, 'energy source 1')].astype(str),
                        'ret_year': pd.to_numeric(
                            d[find_col(c, 'retirement', 'year')],
                            errors='coerce'),
                        'sector': d[find_col(c, 'sector', 'name')].astype(str)
                        if find_col(c, 'sector', 'name') else np.nan}))
        print(y, 'plant rows', sum(len(p) for p in plants if
                                   (p['year'] == y).any()), flush=True)
    pd.concat(plants).to_csv(p_pl, index=False)
    pd.concat(retired).to_csv(p_rt, index=False)
PL = pd.read_csv(p_pl); RT = pd.read_csv(p_rt)
PL['oris'] = PL['oris'].astype(int)
print('Plant panel columns filled (share of rows):',
      PL[['sector', 'regstatus', 'ba']].notna().mean().round(2).to_dict())

# ---------- (A) wind-down by regime ----------
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)
MARKETS = {'PJM', 'MISO', 'ISNE', 'NYIS', 'ERCO', 'SWPP', 'CISO'}
def attrs(fid, T):
    d = PL[PL['oris'] == fid]
    if d.empty:
        return pd.Series(dict(regulated=np.nan, market=np.nan))
    d = d.iloc[(d['year'] - (T - 1)).abs().argsort()].iloc[0]
    rs = str(d['regstatus']).strip().upper()
    sec = str(d['sector']).lower()
    if rs in ('RE', 'NR'):
        reg = rs == 'RE'
    elif sec not in ('nan', ''):
        reg = 'electric utility' in sec
    else:
        reg = np.nan
    ba = str(d['ba']).strip().upper()
    mk = np.nan if ba in ('NAN', '') else (
        ba in MARKETS and not (ba == 'SWPP' and T < 2014))
    return pd.Series(dict(regulated=reg, market=mk))
U = pd.concat([U, U.apply(lambda r: attrs(r.fid, r.last_coal_year), axis=1)],
              axis=1)
print(f'\n(A) {U.regulated.notna().sum()} of {len(U)} exits classified by '
      f'ownership, {U.market.notna().sum()} by market')

def share(d, c, b):
    k = d[b] > 0
    return 1 - d.loc[k, c].sum() / d.loc[k, b].sum()
def med_ratio(d):
    k = (d.heat_t10 > 0) & (d.heat_final > 0)
    return (d.heat_final[k] / d.heat_t10[k]).median()
STATS = {'SO2 share, ten years before': lambda d: share(d, 'so2_final', 'so2_t10'),
         'SO2 share, peak': lambda d: share(d, 'so2_final', 'so2_peak'),
         'heat share, ten years before': lambda d: share(d, 'heat_final', 'heat_t10'),
         'median heat t-1/t-10': med_ratio}
def compare(col, a_lab, b_lab):
    d = U[U[col].notna()].copy(); d[col] = d[col].astype(bool)
    g = {k: v for k, v in d.groupby('fid')}; keys = list(g)
    print(f'\n  {a_lab} vs {b_lab}: {d[col].sum()} vs {(~d[col]).sum()} exits')
    rows = []
    for lab, f in STATS.items():
        ea, eb = f(d[d[col]]), f(d[~d[col]])
        bs = []
        for _ in range(NB):
            s = pd.concat([g[keys[i]] for i in rng.integers(0, len(keys),
                                                            len(keys))])
            bs.append(f(s[s[col]]) - f(s[~s[col]]))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(split=col, stat=lab, a=ea, b=eb, diff=ea - eb,
                         lo=lo, hi=hi))
        print(f'    {lab}: {ea:.2f} vs {eb:.2f}; difference {ea - eb:+.2f} '
              f'[{lo:+.2f}, {hi:+.2f}]', flush=True)
    return rows
R = compare('regulated', 'regulated utility', 'independent producer') \
    + compare('market', 'inside RTO/ISO market', 'outside')
pd.DataFrame(R).to_csv(f'{OUT}/regime_windown.csv', index=False)

# ---------- (B) dating by the official EIA retirement year ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'so2mass',
                         'heatinput', 'primaryfuelinfo'])
for c in ['so2mass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
coal_u = set(a.loc[a['primaryfuelinfo'].fillna('').str.contains('Coal')
                   & (a['heatinput'] > 0), 'uid'])
pc = a[a['uid'].isin(coal_u)].groupby(['facilityid', 'year'])['so2mass'] \
    .sum().unstack(fill_value=0)
full = [f for f, d in a[a['uid'].isin(coal_u)].groupby('facilityid')
        if set(d['uid']) <= set(U['uid'])]
rc = RT[RT['fuel'].str.upper().isin(COAL)].groupby('oris')['ret_year'].max()
last = U.groupby('fid')['last_coal_year'].max()
rows = []
for f in full:
    if f not in rc.index or f not in pc.index or pd.isna(rc[f]):
        continue
    T, R_ = int(last[f]), int(rc[f])
    if R_ < T or T - 11 < pc.columns.min():
        continue
    s = pc.loc[f]
    g = lambda y: s.get(y, 0.0)
    rows.append(dict(fid=f, T=T, R=R_, lag=R_ - T, base_T=g(T - 10),
                     fin_T=g(T - 1), base_R=g(R_ - 10), fin_R=g(R_ - 1)))
Dd = pd.DataFrame(rows)
print(f'\n(B) {len(Dd)} plants whose coal units all exited and that have an '
      f'EIA coal retirement year (lag 0: {(Dd.lag == 0).sum()}, 1 year: '
      f'{(Dd.lag == 1).sum()}, 2+ years: {(Dd.lag >= 2).sum()})')
for lab, b, f in [('last year of coal burning', 'base_T', 'fin_T'),
                  ('official EIA retirement year', 'base_R', 'fin_R')]:
    k = Dd[b] > 0
    print(f'  Dated by {lab}: share of the ten-year SO2 cut before the last '
          f'full year {100 * (1 - Dd.loc[k, f].sum() / Dd.loc[k, b].sum()):.0f}%')
Dd.to_csv(f'{OUT}/dating_eia.csv', index=False)
