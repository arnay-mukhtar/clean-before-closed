# Step 1: complete model predictions for the US plant exit events.
#  (a) primary PM2.5 emissions per plant from the National Emissions
#      Inventory (via EPA ECHO's combined air emissions file, which links
#      NEI facilities to Clean Air Markets plant codes);
#  (b) stack data from EIA-860 (2009-2024) -> Briggs plume height ->
#      InMAP release layer for each plant;
#  (c) predictions = sulfate + nitrate + primary PM2.5, near monitors minus
#      state background, frozen with a fingerprint.
# Uses emissions, stack data, model output, monitor locations and record
# presence only. No PM2.5 values are read.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, re, zipfile, hashlib, requests
import numpy as np, pandas as pd

OUT = f'{DIR}/step1'
os.makedirs(OUT, exist_ok=True)
TPY = 907184.74e6 / (365.25 * 86400)    # µg/s per short ton per year

# ---------- (a) NEI primary PM2.5 per CAMD plant ----------
# Link rule: an EIS (NEI) facility in the same EPA registry record as the
# Clean Air Markets plant; otherwise the nearest EIS facility with PM2.5
# within 3 km of the plant. Pollutant: Primary PM2.5 (filterables and
# condensibles), converted from pounds to short tons.
p_pm = f'{OUT}/nei_pm25_by_oris_v3.csv'
fac = pd.read_csv(f'{DIR}/us_epa_facilities.csv')
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
if not os.path.exists(p_pm):
    z = f'{OUT}/echo_air_emissions.zip'
    if not os.path.exists(z):
        url = ('https://echo.epa.gov/files/echodownloads/'
               'POLL_RPT_COMBINED_EMISSIONS.zip')
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(z, 'wb') as f:
                for ch in r.iter_content(8 << 20):
                    f.write(ch)
    zz = zipfile.ZipFile(z)
    name = [n for n in zz.namelist() if n.lower().endswith('.csv')][0]
    camd, eis = [], []
    for ch in pd.read_csv(zz.open(name), chunksize=2_000_000, dtype=str,
                          encoding='latin-1'):
        ch.columns = [c.upper() for c in ch.columns]
        camd.append(ch[ch['PGM_SYS_ACRNM'] == 'CAMDBS']
                    [['REGISTRY_ID', 'PGM_SYS_ID']].drop_duplicates())
        eis.append(ch[(ch['PGM_SYS_ACRNM'] == 'EIS') & (
            ch['POLLUTANT_NAME'] ==
            'Primary PM2.5 (filterables and condensibles)')])
    camd = pd.concat(camd).drop_duplicates()
    camd['oris'] = pd.to_numeric(camd['PGM_SYS_ID'], errors='coerce')
    eis = pd.concat(eis)
    eis['val'] = pd.to_numeric(eis['ANNUAL_EMISSION'], errors='coerce')
    eis.loc[eis['UNIT_OF_MEASURE'].str.lower().str.contains('pound',
                                                         na=False),
            'val'] /= 2000.0
    reg_pm = (eis.groupby(['REGISTRY_ID', 'REPORTING_YEAR'])['val'].sum()
                 .reset_index())
    # registry coordinates from the ECHO facility export
    ze = f'{OUT}/echo_exporter.zip'
    if not os.path.exists(ze):
        with requests.get('https://echo.epa.gov/files/echodownloads/'
                          'echo_exporter.zip', stream=True,
                          timeout=120) as r:
            r.raise_for_status()
            with open(ze, 'wb') as f:
                for ch in r.iter_content(8 << 20):
                    f.write(ch)
    zx = zipfile.ZipFile(ze)
    nx = [n for n in zx.namelist() if n.lower().endswith('.csv')][0]
    want = {'REGISTRY_ID', 'FAC_LAT', 'FAC_LONG', 'FAC_NAME'}
    ex = pd.read_csv(zx.open(nx), dtype=str, encoding='latin-1',
                     usecols=lambda c: c.upper() in want)
    ex.columns = [c.upper() for c in ex.columns]
    ex['lat'] = pd.to_numeric(ex['FAC_LAT'], errors='coerce')
    ex['lon'] = pd.to_numeric(ex['FAC_LONG'], errors='coerce')
    eis_reg = ex[ex['REGISTRY_ID'].isin(set(reg_pm['REGISTRY_ID']))] \
        .dropna(subset=['lat', 'lon'])
    out = []
    for f in fac.itertuples():
        regs = set(camd.loc[camd['oris'] == f.facilityid, 'REGISTRY_ID'])
        hit = reg_pm[reg_pm['REGISTRY_ID'].isin(regs)]
        link = 'registry'
        if hit.empty and len(eis_reg):
            d = km(eis_reg['lat'].values, eis_reg['lon'].values,
                   f.latitude, f.longitude)
            i = int(np.argmin(d))
            if d[i] <= 3:
                hit = reg_pm[reg_pm['REGISTRY_ID'] ==
                             eis_reg['REGISTRY_ID'].iloc[i]]
                link = f'coords {d[i]:.1f} km'
        for y, v in hit.groupby('REPORTING_YEAR')['val'].sum().items():
            out.append(dict(oris=f.facilityid, year=int(y), pm25_tons=v,
                            link=link))
    pm = pd.DataFrame(out)
    pm.to_csv(p_pm, index=False)
pm = pd.read_csv(p_pm)
pm['oris'] = pm['oris'].astype(int)
print('NEI PM2.5 plant-years:', len(pm), ' plants:', pm['oris'].nunique(),
      ' links:', pm.drop_duplicates('oris')['link'].str[:6]
      .value_counts().to_dict())

# ---------- (b) EIA-860 stack data ----------
p_st = f'{OUT}/eia860_stacks.csv'
def find_col(cols, *keys, avoid=()):
    for c in cols:
        lc = str(c).lower()
        if all(k in lc for k in keys) and not any(a in lc for a in avoid):
            return c
    return None
if not os.path.exists(p_st):
    rows = []
    for y in range(2009, 2025):
        content = None
        for base in ['https://www.eia.gov/electricity/data/eia860/xls/',
                     'https://www.eia.gov/electricity/data/eia860/archive/'
                     'xls/']:
            try:
                r = requests.get(f'{base}eia860{y}.zip', timeout=300)
                if r.ok and r.content[:2] == b'PK':
                    content = r.content; break
            except Exception:
                pass
        if content is None:
            print(y, 'EIA-860 not found'); continue
        zz = zipfile.ZipFile(io.BytesIO(content))
        cands = [n for n in zz.namelist() if re.search(r'enviro|stack', n, re.I)
                 and n.lower().endswith(('.xlsx', '.xls'))]
        got = 0
        for n in cands:
            try:
                book = pd.read_excel(io.BytesIO(zz.read(n)), sheet_name=None,
                                     header=None)
            except Exception as ex:
                print(y, n, 'unreadable', ex); continue
            for sh, raw in book.items():
                txt = raw.head(6).astype(str).apply(' '.join, axis=1).str.lower()
                hdr = [i for i, t in enumerate(txt) if 'height' in t]
                if not hdr:
                    continue
                d = raw.iloc[hdr[0] + 1:].copy()
                names, seen = [], {}
                for c in raw.iloc[hdr[0]].astype(str):
                    seen[c] = seen.get(c, 0) + 1
                    names.append(c if seen[c] == 1 else f'{c}__{seen[c]}')
                d.columns = names        # duplicate headers made unique
                cols = list(d.columns)
                c_pl = find_col(cols, 'plant', 'code') or find_col(cols, 'plant', 'id')
                c_h = find_col(cols, 'height')
                c_v = find_col(cols, 'velocity', '100') or find_col(cols, 'velocity')
                c_t = (find_col(cols, 'temp', '100') or
                       find_col(cols, 'temp', avoid=('summer', 'winter', 'measured')))
                c_a = find_col(cols, 'area') or find_col(cols, 'diam')
                c_q = find_col(cols, 'rate', '100') or find_col(cols, 'flow')
                if not (c_pl and c_h):
                    continue
                o = pd.DataFrame({
                    'oris': pd.to_numeric(d[c_pl], errors='coerce'),
                    'height_ft': pd.to_numeric(d[c_h], errors='coerce'),
                    'vel_fps': pd.to_numeric(d[c_v], errors='coerce') if c_v else np.nan,
                    'temp_f': pd.to_numeric(d[c_t], errors='coerce') if c_t else np.nan,
                    'area_or_diam': pd.to_numeric(d[c_a], errors='coerce') if c_a else np.nan,
                    'is_diam': bool(c_a and 'diam' in str(c_a).lower()),
                    'flow_cfm': pd.to_numeric(d[c_q], errors='coerce') if c_q else np.nan,
                    'year': y})
                o = o.dropna(subset=['oris', 'height_ft'])
                rows.append(o); got += len(o)
        print(y, 'stack rows:', got, flush=True)
    st = pd.concat(rows)
    st.to_csv(p_st, index=False)
st = pd.read_csv(p_st)
st['oris'] = st['oris'].astype(int)

# Briggs final plume rise, neutral conditions, wind 5 m/s, ambient 288 K
def plume_height(r, u=5.0, Ta=288.0, g=9.81):
    h = r['height_ft'] * 0.3048
    v = r['vel_fps'] * 0.3048
    Ts = (r['temp_f'] - 32) * 5 / 9 + 273.15
    if r['is_diam']:
        dm = r['area_or_diam'] * 0.3048
    else:
        dm = 2 * np.sqrt(r['area_or_diam'] * 0.092903 / np.pi)
    if not (np.isfinite(v) and np.isfinite(Ts) and np.isfinite(dm)) \
            or Ts <= Ta or v <= 0 or dm <= 0:
        return np.nan, h
    F = g * v * dm ** 2 * (Ts - Ta) / (4 * Ts)
    xs = 14 * F ** 0.625 if F < 55 else 34 * F ** 0.4
    dh = 1.6 * F ** (1 / 3) * (3.5 * xs) ** (2 / 3) / u
    return h + dh, h
ph = st.apply(lambda r: pd.Series(plume_height(r),
                                  index=['eff_m', 'stack_m']), axis=1)
st = pd.concat([st, ph], axis=1)
st['w'] = st['flow_cfm'].fillna(1.0).clip(lower=1.0)

def plant_height(fid, year):
    s = st[(st['oris'] == fid)]
    if s.empty:
        return np.nan, np.nan, 'none'
    # stack data from the latest year before the event, else earliest after
    before = s[s['year'] <= year - 1]
    s = s[s['year'] == (before['year'].max() if len(before) else
                        s['year'].min())]
    ok = s.dropna(subset=['eff_m'])
    if ok.empty:
        return np.nan, s['stack_m'].max(), 'stack only'
    eff = np.average(ok['eff_m'], weights=ok['w'])
    return eff, ok['stack_m'].max(), 'full'

# ---------- (c) predictions ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput', 'primaryfuelinfo',
                         'pmcontrolinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
CH = a[a['coal']].groupby(['facilityid', 'year'])['heatinput'].sum()
def pmcat(s):
    s = str(s).lower()
    if 'baghouse' in s or 'fabric' in s:
        return 'fabric filter'
    if 'electrostatic' in s:
        return 'ESP'
    if s in ('', 'nan'):
        return 'none reported'
    return 'other'
cat = (a[a['coal']].assign(c=a['pmcontrolinfo'].map(pmcat))
       .groupby(['facilityid', 'year'])['c']
       .agg(lambda x: x.value_counts().index[0]))
# PM2.5 rate per unit of coal heat input, by plant and NEI year
pm['coal_heat'] = [CH.get((o, y), 0.0) for o, y in zip(pm['oris'],
                                                       pm['year'])]
pm = pm[pm['coal_heat'] > 0].copy()
pm['rate'] = pm['pm25_tons'] / pm['coal_heat']
pm['cat'] = [cat.get((o, y), 'none reported') for o, y in
             zip(pm['oris'], pm['year'])]
cat_med = pm.groupby('cat')['rate'].median()
print('Median PM2.5 rate by control type (lb per mmBtu):',
      (cat_med * 2000).round(4).to_dict())

E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
E['cell'] = E['fid'].map(fc.set_index('facilityid')['cell']).fillna(-1) \
             .astype(int)
rec = []
for r in E.itertuples():
    hb = CH.get((r.fid, r.event_year - 1), 0.0)
    ha = CH.get((r.fid, r.event_year + 1), 0.0)
    p = pm[pm['oris'] == r.fid]
    pb = p[p['year'] <= r.event_year - 1]
    use = pb if len(pb) else p
    if len(use):
        rate = use['rate'].median()          # median over NEI years
        pm_src = 'NEI ' + use['link'].iloc[0][:8]
    else:
        c = cat.get((r.fid, r.event_year - 1), 'none reported')
        rate = cat_med.get(c, cat_med.median())
        pm_src = f'imputed ({c})'
    pm_drop = max(rate * (hb - ha), 0.0)
    eff, stk, how = plant_height(r.fid, r.event_year)
    if np.isfinite(eff):
        layer = 2 if eff > 379 else 1
        lay_src = 'plume rise'
    else:
        layer, lay_src = 2, 'default high (no stack data)'
    rec.append(dict(fid=r.fid, event_year=r.event_year, coal_heat_before=hb,
                    coal_heat_after=ha, pm25_rate_lb_mmbtu=rate * 2000,
                    pm25_drop_tpy=pm_drop, pm25_source=pm_src,
                    stack_m=stk, eff_height_m=eff, height_data=how,
                    layer=layer, layer_source=lay_src))
X = E.merge(pd.DataFrame(rec), on=['fid', 'event_year'])

R = np.load(f'{DIR}/isrm_rows_allcoal.npz')
rcells = {c: i for i, c in enumerate(R['cells'])}
rows = {k: R[k] for k in R.files if k != 'cells'}
G = np.load(f'{DIR}/isrm_grid.npz')
from pyproj import Transformer
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, Ee, S, N = G['W'], G['E'], G['S'], G['N']
def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    h = np.where((W <= x) & (x < Ee) & (S <= y) & (y < N))[0]
    return int(h[0]) if len(h) else -1

frames = [pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                      usecols=['site', 'ym', 'lat', 'lon', 'state', 'ndays'])
          for y in range(2000, 2026)]
q = pd.concat(frames)
q = q[q['ndays'] >= 8]
q['yr'] = q['ym'] // 100
sites = q.groupby('site')[['lat', 'lon', 'state']].first()
n_yr = q.groupby(['site', 'yr']).size()
yrs = n_yr[n_yr >= 8].reset_index().groupby('site')['yr'].apply(set)
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
sites['cell'] = [cell_of(la, lo) for la, lo in sites[['lat', 'lon']].values]
Sx = sites[['lat', 'lon']].values
mind = np.full(len(Sx), np.inf)
for la, lo in fc[['latitude', 'longitude']].values:
    mind = np.minimum(mind, km(Sx[:, 0], Sx[:, 1], la, lo))
sites['bg'] = (mind > 50) & (sites['cell'] >= 0)

out = []
for r in X.itertuples():
    if r.cell not in rcells:
        continue
    j = rcells[r.cell]
    b, a_ = r.event_year - 1, r.event_year + 1
    d = km(Sx[:, 0], Sx[:, 1], r.lat, r.lon)
    near = [s for s in sites.index[d <= 25]
            if {b, a_} <= yrs.get(s, set()) and sites.at[s, 'cell'] >= 0]
    o = dict(fid=r.fid, name=r.name, state=r.state, event_year=r.event_year,
             so2_drop=r.so2_drop, nox_drop=r.nox_drop,
             pm25_drop_tpy=r.pm25_drop_tpy, pm25_source=r.pm25_source,
             pm25_rate_lb_mmbtu=r.pm25_rate_lb_mmbtu,
             stack_m=r.stack_m, eff_height_m=r.eff_height_m,
             layer=r.layer, layer_source=r.layer_source)
    pmd = 0.0 if not np.isfinite(r.pm25_drop_tpy) else r.pm25_drop_tpy
    for L, tag in [(r.layer, 'main'), (2, 'high'), (1, 'low')]:
        comp = {'so4': rows[f'pSO4_L{L}'][j].astype('f8') * max(r.so2_drop, 0),
                'no3': rows[f'pNO3_L{L}'][j].astype('f8') * max(r.nox_drop, 0),
                'ppm': rows[f'PrimaryPM25_L{L}'][j].astype('f8') * pmd}
        vals = {k: [] for k in comp}
        for s in near:
            st_ = sites.at[s, 'state']
            bg = [x for x in sites.index[(sites['state'] == st_) & sites['bg']]
                  if {b, a_} <= yrs.get(x, set())]
            if not bg:
                continue
            bc = sites.loc[bg, 'cell'].values
            mc = sites.at[s, 'cell']
            for k, v in comp.items():
                vals[k].append((v[mc] - v[bc].mean()) * TPY)
        o['n_near_bg'] = len(vals['so4'])
        for k in comp:
            o[f'{k}_{tag}'] = np.mean(vals[k]) if vals[k] else np.nan
        o[f'pred_{tag}'] = (o[f'so4_{tag}'] + o[f'no3_{tag}']
                            + o[f'ppm_{tag}'])
        o[f'pred_noppm_{tag}'] = o[f'so4_{tag}'] + o[f'no3_{tag}']
    out.append(o)
P = pd.DataFrame(out)
path = f'{DIR}/model_predictions_us_events_v2.csv'
P.to_csv(path, index=False)
h = hashlib.sha256(open(path, 'rb').read()).hexdigest()

pd.set_option('display.width', 230)
m = P[P['n_near_bg'] > 0]
pr = m[m['so2_drop'] >= 5000]
print(f'\nEvents with monitor and background: {len(m)}; primary sample '
      f'(SO2 drop >= 5,000 t): {len(pr)}')
print('Stack data:', pr['layer_source'].value_counts().to_dict())
print('PM2.5 source:', pr['pm25_source'].value_counts().to_dict())
print('PM2.5 rate used (lb/mmBtu), quartiles:',
      pr['pm25_rate_lb_mmbtu'].quantile([.25, .5, .75]).round(4).tolist())
print('Effective plume height, primary sample (m), quartiles:',
      pr['eff_height_m'].quantile([.25, .5, .75]).round(0).tolist())
print('Release layer used:', pr['layer'].map({1: 'low', 2: 'high'})
      .value_counts().to_dict())
print('Primary PM2.5 drop (short tons/yr), quartiles:',
      pr['pm25_drop_tpy'].quantile([.25, .5, .75]).round(0).tolist())
print('\nMean predicted fall, primary sample, µg/m³ (near minus background):')
for tag in ['main', 'high', 'low']:
    print(f'  {tag:4s}: total {pr[f"pred_{tag}"].mean():.3f} = sulfate '
          f'{pr[f"so4_{tag}"].mean():.3f} + nitrate {pr[f"no3_{tag}"].mean():.3f}'
          f' + primary PM2.5 {pr[f"ppm_{tag}"].mean():.3f}')
print('\nLargest predicted falls (main):')
print(pr.sort_values('pred_main', ascending=False)
      [['name', 'event_year', 'so2_drop', 'pm25_drop_tpy', 'eff_height_m',
        'layer', 'pred_main', 'ppm_main']].round(3).head(12)
      .to_string(index=False))
print('\nSHA-256 of model_predictions_us_events_v2.csv:', h)
