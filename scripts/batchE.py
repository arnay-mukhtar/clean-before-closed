# POST-REGISTRATION: registered SECONDARY analyses, part 2 (batch E,
# 10 Nov 2026). Earth Engine needed (project coal-closures-air).
#  (A) Downwind design, HOURLY (registration Q3 item 4): hourly AQS PM2.5
#      at monitors within 25 km of the primary events that have an hourly
#      monitor (10 at filing), hourly ERA5-Land wind at the plant; same
#      statistic as the daily design with the hourly rules used for the
#      frozen placebo (100+ downwind and 100+ upwind hours per site-year).
#  (B) Secondary NO2 test: events from 2019 in the US and Europe with a NOx
#      drop of at least 1,000 tons; change in the TROPOMI NO2 contrast (log
#      mean column within 10 km minus 25-50 km) from the year before to the
#      year after, against placebo changes at never-exit plants.
#  (C) Q5 Kazakhstan fallback: unit output is not published for the two
#      Kazakh exits in the GEM tracker (Aksu unit 7, 2024; Topar unit 3,
#      2025), so the registered satellite activity measure is reported:
#      the NO2 contrast in the year before retirement relative to its
#      pre-period mean (labelled as a station-level activity measure).
DIR = '/content/drive/MyDrive/coal_closures'
exec(open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0])
# names W, Ee, S, N, G, E, U, a, ex, fc, B, rng, ctrl, tr are used by the
# helpers above: do not reuse them below
import io, time, zipfile, requests, hashlib, ee
try:
    ee.Initialize(project='coal-closures-air')
except Exception:
    ee.Authenticate(); ee.Initialize(project='coal-closures-air')
saved = []
def save(df, name):
    p = f'{OUT}/{name}'; df.to_csv(p, index=False); saved.append(p)
fpr2 = pd.read_csv(f'{DIR}/fingerprints.csv').set_index('file')['sha256']
for f in ['downwind_hourly_placebo_draws.csv', 'hourly_event_coverage.csv',
          'aqs_hourly_monitor_years.csv']:
    h = hashlib.sha256(open(f'{DIR}/{f}', 'rb').read()).hexdigest()
    print(f'  {f}: {"matches" if h == fpr2.get(f) else "DIFFERS FROM"} '
          'the registered fingerprint')

def bearing(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = np.sin(o2 - o1) * np.cos(a2)
    y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(o2 - o1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def report(lab, d, ycol, sd_unit, scale_col=None, alt=None):
    y = d[ycol].values; m = y.mean()
    sc = d[scale_col].values if scale_col else np.ones(len(d))
    s_ = pd.Series(sc).groupby(d['fid'].values).sum()
    se = sd_unit * np.sqrt((s_ ** 2).sum()) / len(d)
    bs = cluster_boot(d, lambda z: z[ycol].mean())
    p_pl = 2 * (1 - Phi(abs(m / se)))
    p_wb = wild_p(y, d['fid'].values, 0.0)[1] if d['fid'].nunique() > 1 else np.nan
    p_bs = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    r = dict(analysis=lab, events=len(d), plants=d['fid'].nunique(), mean=m,
             se_placebo=se, lo_placebo=m - 1.96 * se, hi_placebo=m + 1.96 * se,
             lo_boot=np.percentile(bs, 2.5), hi_boot=np.percentile(bs, 97.5),
             p0_placebo=p_pl, p0_wild=p_wb, p0_boot=p_bs,
             p0_primary=np.nanmax([p_pl, p_wb, p_bs]))
    if alt is not None:
        q_pl = Phi((m - alt) / se)
        q_wb = wild_p(y, d['fid'].values, alt, True)[1]
        q_bs = (bs >= alt).mean()
        r.update(p_below_alt_primary=max(q_pl, q_wb, q_bs))
    print(f"  {lab}: {len(d)} events at {d['fid'].nunique()} plants; mean "
          f"{m:.4f}; placebo 95% [{m - 1.96 * se:.4f}, {m + 1.96 * se:.4f}]; "
          f"bootstrap [{r['lo_boot']:.4f}, {r['hi_boot']:.4f}]; p(0): placebo "
          f"{p_pl:.3f}, wild {p_wb:.3f}, bootstrap {p_bs:.3f} -> primary "
          f"{r['p0_primary']:.3f}"
          + (f"; p(mean >= {alt}) primary {r['p_below_alt_primary']:.4f}"
             if alt is not None else ''))
    return r

def get_region(col, pts, s0, s1):
    for tries in range(5):
        try:
            reg = col.filterDate(s0, s1).getRegion(pts, 11132).getInfo()
            return pd.DataFrame(reg[1:], columns=reg[0])
        except Exception as e_:
            print('  retry', s0, str(e_)[:60]); time.sleep(15)
    raise SystemExit(f'Earth Engine failed for {s0}. Send me this.')

SUM = []
F = pd.read_csv(f'{OUT}/events_pm25_35km.csv')

# ======================================================================
# (A) DOWNWIND, HOURLY
# ======================================================================
print('\n' + '=' * 70 + '\n(A) Downwind design, hourly PM2.5\n' + '=' * 70)
hm = pd.read_csv(f'{DIR}/aqs_hourly_monitor_years.csv')
hm = hm[hm['n'] >= 2000]
hy = hm.groupby('site')['Year'].apply(set)
hst = hm.groupby('site')[['lat', 'lon']].first()
Hx = hst[['lat', 'lon']].values
pairs_e = []
for r in F.itertuples():
    d = km(Hx[:, 0], Hx[:, 1], r.lat, r.lon)
    for i in np.where(d <= 25)[0]:
        s_ = hst.index[i]
        if {r.event_year - 1, r.event_year + 1} <= hy.get(s_, set()):
            pairs_e.append(dict(fid=r.fid, event_year=r.event_year, site=s_,
                                bearing=bearing(r.lat, r.lon, Hx[i, 0],
                                                Hx[i, 1])))
pairs_e = pd.DataFrame(pairs_e)
evh = pairs_e[['fid', 'event_year']].drop_duplicates()
print(f'Primary events with an hourly monitor within 25 km in both years: '
      f'{len(evh)} (10 at filing)')
need = {}
for r in pairs_e.itertuples():
    for y in (r.event_year - 1, r.event_year + 1):
        need.setdefault(y, set()).add(r.site)
# hourly wind at the event plants
p_w = f'{DIR}/wind_hourly_events.csv'
if not os.path.exists(p_w):
    col = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select(
        ['u_component_of_wind_10m', 'v_component_of_wind_10m'])
    pl = F[F['fid'].isin(evh['fid'])].drop_duplicates('fid')
    parts = []
    for y in sorted(need):
        fids = evh.loc[(evh.event_year - 1 == y) | (evh.event_year + 1 == y),
                       'fid'].unique()
        sub = pl[pl['fid'].isin(fids)]
        pts = ee.Geometry.MultiPoint(sub[['lon', 'lat']].values.tolist())
        for m0 in range(1, 13):
            s0 = f'{y}-{m0:02d}-01'
            s1 = f'{y + (m0 == 12)}-{m0 % 12 + 1:02d}-01'
            w = get_region(col, pts, s0, s1).dropna()
            px = w[['longitude', 'latitude']].drop_duplicates().values
            for f_ in sub.itertuples():
                dd = (px[:, 0] - f_.lon) ** 2 + (px[:, 1] - f_.lat) ** 2
                j = int(np.argmin(dd))
                if dd[j] > 0.2 ** 2:
                    continue
                ww = w[(w.longitude == px[j, 0]) & (w.latitude == px[j, 1])]
                parts.append(pd.DataFrame(dict(
                    fid=f_.fid, hour=pd.to_datetime(ww['time'], unit='ms')
                    .dt.strftime('%Y-%m-%d %H'),
                    u=ww['u_component_of_wind_10m'].values,
                    v=ww['v_component_of_wind_10m'].values)))
        print(y, 'hourly wind done', flush=True)
    pd.concat(parts).drop_duplicates(['fid', 'hour']).to_csv(p_w, index=False)
wh = pd.read_csv(p_w)
wh['to'] = (np.degrees(np.arctan2(wh['u'], wh['v'])) + 360) % 360
# hourly PM2.5 at the event monitors
p_h = f'{DIR}/pm25_hourly_events.csv'
if not os.path.exists(p_h):
    parts = []
    for y in sorted(need):
        for tries in range(4):
            try:
                rq = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/'
                                  f'hourly_88101_{y}.zip', timeout=1800)
                rq.raise_for_status(); break
            except Exception as e_:
                print('  retry', y, str(e_)[:60]); time.sleep(30)
        z = zipfile.ZipFile(io.BytesIO(rq.content)); got = 0
        for ch in pd.read_csv(z.open(z.namelist()[0]), chunksize=2_000_000,
                              low_memory=False,
                              usecols=['State Code', 'County Code', 'Site Num',
                                       'Date GMT', 'Time GMT',
                                       'Sample Measurement']):
            ch['site'] = (ch['State Code'].astype(str) + '-'
                          + ch['County Code'].astype(str) + '-'
                          + ch['Site Num'].astype(str))
            ch = ch[ch['site'].isin(need[y])]
            ch['hour'] = ch['Date GMT'] + ' ' + ch['Time GMT'].str[:2]
            parts.append(ch.groupby(['site', 'hour'])['Sample Measurement']
                         .mean().reset_index())
            got += len(ch)
        del z, rq
        print(y, 'hourly PM2.5 rows kept', got, flush=True)
    pd.concat(parts).rename(columns={'Sample Measurement': 'pm'}) \
        .to_csv(p_h, index=False)
ph = pd.read_csv(p_h)
ph = ph[ph['pm'] > 0].copy()
ph['l'] = np.log(ph['pm']); ph['yr'] = ph['hour'].str[:4].astype(int)

def du_hourly(pairs, pm, wind, key):
    X = pairs.merge(pm, on='site').merge(wind[[key, 'hour', 'to']],
                                         on=[key, 'hour'])
    diff = np.abs((X['to'] - X['bearing'] + 180) % 360 - 180)
    X['dir'] = np.where(diff <= 45, 'down', np.where(diff >= 135, 'up', 'side'))
    g = X[X['dir'] != 'side'].groupby([key, 'site', 'yr', 'dir'])
    s = g['l'].agg(['mean', 'size']).unstack('dir')
    s = s[(s[('size', 'down')] >= 100) & (s[('size', 'up')] >= 100)]
    out = (s[('mean', 'down')] - s[('mean', 'up')]).rename('du').reset_index()
    return out, X.groupby(['site', 'yr'])['pm'].mean(), (X['dir'] == 'down').mean()

DU, LVh, share_h = du_hourly(pairs_e, ph, wh, 'fid')
DU = DU.set_index(['fid', 'site', 'yr'])['du']
ev = []
for r in evh.merge(F[['fid', 'event_year', 'name', 'pred_main']]).itertuples():
    b_, a_ = r.event_year - 1, r.event_year + 1
    v, l = [], []
    for s_ in pairs_e.loc[(pairs_e.fid == r.fid)
                          & (pairs_e.event_year == r.event_year), 'site']:
        if (r.fid, s_, b_) in DU.index and (r.fid, s_, a_) in DU.index:
            v.append(DU[(r.fid, s_, b_)] - DU[(r.fid, s_, a_)])
            l.append(LVh[(s_, b_)])
    if v:
        ev.append(dict(fid=r.fid, name=r.name, event_year=r.event_year,
                       n_mon=len(v), dlog=np.mean(v), level=np.mean(l),
                       pred_main=r.pred_main))
DH = pd.DataFrame(ev)
print(f'Events with 100+ downwind and 100+ upwind hours in both years: '
      f'{len(DH)}; share of downwind hours {share_h:.2f}')
# control adjustment from the never-exit plants' hourly data (pre-filing)
mon = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                             usecols=['site', 'lat', 'lon'])
                 for y in range(2003, 2020)]).drop_duplicates('site')
Mx = mon[['lat', 'lon']].values
dex = np.full(len(Mx), np.inf)
for la, lo in ex[['latitude', 'longitude']].values:
    dex = np.minimum(dex, km(Mx[:, 0], Mx[:, 1], la, lo))
pairs_c = []
for c in ctrl.itertuples():
    d = km(Mx[:, 0], Mx[:, 1], c.latitude, c.longitude)
    for i in np.where((d <= 25) & (dex > 25))[0]:
        pairs_c.append(dict(facilityid=c.facilityid, site=mon['site'].iloc[i],
                            bearing=bearing(c.latitude, c.longitude,
                                            Mx[i, 0], Mx[i, 1])))
pairs_c = pd.DataFrame(pairs_c)
pc = pd.read_csv(f'{DIR}/pm25_hourly_controls.csv')
pc = pc[pc['pm'] > 0].copy()
pc['l'] = np.log(pc['pm']); pc['yr'] = pc['hour'].str[:4].astype(int)
wc = pd.read_csv(f'{DIR}/wind_hourly_controls.csv')
wc['to'] = (np.degrees(np.arctan2(wc['u'], wc['v'])) + 360) % 360
DUc, _, _ = du_hourly(pairs_c, pc, wc, 'facilityid')
Dc = {}
for (f, si), gg in DUc.groupby(['facilityid', 'site']):
    gg = gg.set_index('yr')['du']
    for y in range(2004, 2019):
        if y - 1 in gg.index and y + 1 in gg.index:
            Dc.setdefault((f, y), []).append(gg[y - 1] - gg[y + 1])
Dc = pd.Series({k: np.mean(v) for k, v in Dc.items()})
ym_h = Dc.groupby(level=1).mean(); pool_h = Dc.mean()
sd_h = pd.read_csv(f'{DIR}/downwind_hourly_placebo_draws.csv')['d'].std()
print(f'Placebo: {len(Dc)} plant-years at '
      f'{Dc.index.get_level_values(0).nunique()} never-exit plants, pooled '
      f'mean {pool_h:.4f} log; frozen draws SD {sd_h:.4f} log')
if len(DH):
    DH['adj'] = DH['event_year'].map(ym_h).fillna(pool_h)
    DH['fall_raw'] = DH['dlog'] * DH['level']
    DH['fall'] = (DH['dlog'] - DH['adj']) * DH['level']
    SUM.append(report('Downwind hourly, control-adjusted (ug/m3)', DH, 'fall',
                      sd_h, 'level', 1.1))
    SUM.append(report('Downwind hourly, unadjusted (ug/m3)', DH, 'fall_raw',
                      sd_h, 'level'))
    print(f'  rough downwind-hour prediction: '
          f'{DH.pred_main.mean() / share_h:.2f} ug/m3')
    save(DH, 'downwind_hourly_events.csv')

# ======================================================================
# (B) SECONDARY NO2 TEST and (C) KAZAKHSTAN FALLBACK (TROPOMI)
# ======================================================================
print('\n' + '=' * 70 + '\n(B) Secondary NO2 test (TROPOMI)\n' + '=' * 70)
EVu = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
EVu = EVu[(EVu['event_year'].between(2019, 2024)) & (EVu['nox_drop'] >= 1000)]
EVu = EVu.assign(region='US', t=EVu['event_year'])[['fid', 'name', 'region',
                                                    'lat', 'lon', 't',
                                                    'nox_drop']]
# Europe: site-year events, last coal year 2019-2022, site NOx drop >= 1,000 t
d5 = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv', low_memory=False,
                 usecols=['LCPInspireId', 'Latitude', 'Longitude',
                          'countryName', 'reportingYear', 'featureType',
                          'unit', 'featureValue'])
loc = d5.groupby('LCPInspireId')[['Latitude', 'Longitude']].first()
EUx = pd.read_csv(f'{DIR}/eu_unit_exits.csv')
EUx = (EUx[EUx['last_coal_year'].between(2019, 2022)]
       .sort_values('unit_mw_th', ascending=False)
       .drop_duplicates(['site', 'last_coal_year']))
EUx = EUx[EUx['site_nox_drop'] >= 1000].join(loc, on='lcp')
EVe = pd.DataFrame(dict(fid=['EU' + str(s) for s in EUx['site']],
                        name=EUx['name'], region='EU', lat=EUx['Latitude'],
                        lon=EUx['Longitude'], t=EUx['last_coal_year'],
                        nox_drop=EUx['site_nox_drop']))
print(f'Events: US {len(EVu)}, Europe {len(EVe)}')
# placebo plants: US coal burned every year 2017-2025, no unit exit, more
# than 50 km from any US exit 2017-2025; Europe coal sites burning coal in
# every reported year 2016-2023 with no unit exit, more than 50 km from any
# European exit 2017-2023
a3 = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                 usecols=['facilityid', 'year', 'heatinput', 'primaryfuelinfo'])
a3['heatinput'] = pd.to_numeric(a3['heatinput'], errors='coerce').fillna(0)
cy3 = a3[a3['primaryfuelinfo'].fillna('').str.contains('Coal')
         & (a3['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
EUS = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
recent = EUS[EUS['event_year'].between(2017, 2025)]
cand = fc[fc['facilityid'].isin([f for f, ys in cy3.items()
                                 if set(range(2017, 2026)) <= ys
                                 and f not in set(EUS['fid'])])]
keep = [r for r in cand.itertuples()
        if km(recent['lat'].values, recent['lon'].values, r.latitude,
              r.longitude).min() > 50]
PLu = pd.DataFrame(dict(fid=[r.facilityid for r in keep], region='US',
                        lat=[r.latitude for r in keep],
                        lon=[r.longitude for r in keep]))
d5['featureValue'] = pd.to_numeric(d5['featureValue'], errors='coerce')
d5['key'] = np.where(d5['featureType'] == 'LCPCharacteristics', d5['unit'],
                     d5['featureType'])
wv = d5.pivot_table(index=['LCPInspireId', 'reportingYear'], columns='key',
                    values='featureValue', aggfunc='sum').fillna(0)
solid = (wv.get('Coal', 0) + wv.get('Lignite', 0)).rename('solid')
burn = solid[solid > 0].reset_index().groupby('LCPInspireId')['reportingYear'] \
    .apply(set)
allx = pd.read_csv(f'{DIR}/eu_unit_exits.csv')
exited = set(allx['lcp'])
eu_rec = allx[allx['last_coal_year'].between(2017, 2023)].join(loc, on='lcp')
ok_ = [l for l, ys in burn.items() if set(range(2016, 2024)) <= ys
       and l not in exited]
cpl = loc.loc[ok_]
cpl = cpl[[km(eu_rec['Latitude'].values, eu_rec['Longitude'].values,
              r.Latitude, r.Longitude).min() > 50 for r in cpl.itertuples()]]
# one placebo per 3 km site
cpl = cpl.round(2).drop_duplicates()
PLe = pd.DataFrame(dict(fid=['EUP' + str(i) for i in range(len(cpl))],
                        region='EU', lat=cpl['Latitude'].values,
                        lon=cpl['Longitude'].values))
print(f'Placebo plants: US {len(PLu)}, Europe {len(PLe)}')
KZ = pd.DataFrame(dict(fid=['KZ_Aksu', 'KZ_Topar'], region='KZ',
                       name=['Aksu (unit 7, 300 of 2,450 MW, retired 2024)',
                             'Topar (unit 3, 100 of about 700 MW, retired 2025)'],
                       lat=[52.120048, 49.51333], lon=[76.875002, 72.79861],
                       t=[2024, 2025]))
PTS = pd.concat([EVu, EVe, PLu, PLe, KZ], ignore_index=True)
PTS = PTS.drop_duplicates('fid').reset_index(drop=True)
feats = []
for i, r in PTS.iterrows():
    pt = ee.Geometry.Point([float(r.lon), float(r.lat)])
    feats.append(ee.Feature(pt.buffer(10000), {'k': i, 'zone': 'near'}))
    feats.append(ee.Feature(pt.buffer(50000).difference(pt.buffer(25000), 100),
                            {'k': i, 'zone': 'far'}))
FCo = ee.FeatureCollection(feats)
no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2') \
    .select('tropospheric_NO2_column_number_density')
p_n = f'{DIR}/tropomi_no2_annual.csv'
if not os.path.exists(p_n):
    rows = []
    periods = [(str(y), f'{y}-01-01', f'{y + 1}-01-01') for y in range(2019, 2026)]
    periods += [(f'{y}H2', f'{y}-07-01', f'{y + 1}-01-01') for y in (2018, 2020)]
    for lab, s0, s1 in periods:
        img = no2.filterDate(s0, s1).mean()
        for tries in range(5):
            try:
                res = img.reduceRegions(FCo, ee.Reducer.mean(), 1113,
                                        tileScale=4).getInfo(); break
            except Exception as e_:
                print('  retry', lab, str(e_)[:80]); time.sleep(20)
        for f_ in res['features']:
            p_ = f_['properties']
            rows.append(dict(k=p_['k'], zone=p_['zone'], period=lab,
                             no2=p_.get('mean')))
        print(lab, 'done', flush=True)
    pd.DataFrame(rows).to_csv(p_n, index=False)
NO = pd.read_csv(p_n)
NO = NO.pivot_table(index=['k', 'period'], columns='zone', values='no2')
NO['C'] = np.log(NO['near']) - np.log(NO['far'])
NO['X'] = NO['near'] - NO['far']
C_ = NO['C']
def cstat(k, t):
    if t - 1 == 2018:
        a1, a2 = (k, '2018H2'), (k, '2020H2')
    else:
        a1, a2 = (k, str(t - 1)), (k, str(t + 1))
    if a1 in C_.index and a2 in C_.index:
        return C_[a1] - C_[a2]
    return np.nan
PTS['k'] = PTS.index
EVall = PTS[PTS.region.isin(['US', 'EU']) & PTS['t'].notna()].copy()
EVall['dC'] = [cstat(r.k, int(r.t)) for r in EVall.itertuples()]
PL = []
for r in PTS[PTS['t'].isna() & PTS.region.isin(['US', 'EU'])].itertuples():
    for y in range(2019, 2025 if r.region == 'US' else 2023):
        v = cstat(r.k, y)
        if np.isfinite(v):
            PL.append(dict(fid=r.fid, region=r.region, year=y, dC=v))
PL = pd.DataFrame(PL)
print(f'Placebo changes: {len(PL)} at {PL.fid.nunique()} plants; mean '
      f'{PL.dC.mean():.4f}, SD {PL.dC.std():.4f}')
ymp = PL.groupby(['region', 'year'])['dC'].mean()
poolp = PL.groupby('region')['dC'].mean()
EVall = EVall[EVall['dC'].notna()].copy()
EVall['fid'] = EVall['fid'].astype(str)          # US ids and EU site ids
PL['fid'] = PL['fid'].astype(str)
EVall['adj'] = [ymp.get((r.region, int(r.t)), poolp[r.region])
                for r in EVall.itertuples()]
EVall['fall'] = EVall['dC'] - EVall['adj']
sdp = PL['dC'].std()
for lab, sub in [('NO2 contrast, US and Europe', EVall),
                 ('NO2 contrast, US', EVall[EVall.region == 'US']),
                 ('NO2 contrast, Europe', EVall[EVall.region == 'EU'])]:
    if len(sub) >= 3:
        SUM.append(report(lab + ' (log points)', sub, 'fall', sdp))
b = slope0(EVall['nox_drop'] / 1e3, EVall['fall'])
bs = cluster_boot(EVall, lambda z: slope0(z['nox_drop'] / 1e3, z['fall']))
print(f'  fall per 1,000 t NOx drop: {b:.4f} '
      f'[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}] '
      '(exploratory dose check)')
save(EVall, 'no2_secondary_events.csv'); save(PL, 'no2_secondary_placebo.csv')

print('\n' + '=' * 70 + '\n(C) Kazakhstan, satellite activity measure '
      '(station level)\n' + '=' * 70)
X_ = NO['X']
for r in PTS[PTS.region == 'KZ'].itertuples():
    t = int(r.t)
    pre = [y for y in range(2019, t - 1) if (r.k, str(y)) in C_.index]
    if (r.k, str(t - 1)) not in C_.index or not pre:
        print(f'  {r.name}: not enough TROPOMI years'); continue
    dC = C_[(r.k, str(t - 1))] - np.mean([C_[(r.k, str(y))] for y in pre])
    rx = X_[(r.k, str(t - 1))] / np.mean([X_[(r.k, str(y))] for y in pre])
    path = ', '.join(f'{y}: {C_[(r.k, str(y))]:.3f}'
                     for y in range(2019, 2026) if (r.k, str(y)) in C_.index)
    print(f'  {r.name}: contrast in year t-1 ({t - 1}) minus its {pre[0]}-'
          f'{pre[-1]} mean {dC:+.3f} log points; near-minus-far column excess '
          f't-1 relative to pre-period {rx:.2f}\n    contrast by year: {path}')
    SUM.append(dict(analysis=f'Q5 Kazakhstan satellite activity: {r.name}',
                    mean=dC, ratio_excess=rx))

save(pd.DataFrame(SUM), 'batchE_summary.csv')
print('\nSHA-256:')
for p in saved:
    print(f'  {os.path.basename(p)}  '
          f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
