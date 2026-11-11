# POST-REGISTRATION: registered SECONDARY analysis, part 3 (batch F,
# 10 Nov 2026): the European causal arm.
#  Registration: large European unit exits (site SO2 or NOx drop of at least
#  1,000 tonnes, last coal year 2017-2022; 37 at filing); outcomes from EEA
#  regulatory stations within 25 km (PM2.5, SO2, NO2); same event statistic
#  (relative to background, year before minus year after); tested against
#  zero only. Data: EEA Air Quality Download Service (verified E1a, plus
#  up-to-date E2a for 2023), station metadata from the EEA metadata extract.
#  Choices the registration leaves open (stated in the paper): exits dated
#  by last coal year (the statistic excludes the exit year); background =
#  up to 40 same-country stations more than 50 km from any coal site
#  (background station type where recorded), chosen with a fixed seed;
#  placebo = never-exit European coal sites.
#  Also: Europe Q2 valuation rerun including Serbia (omitted in batch D).
DIR = '/content/drive/MyDrive/coal_closures'
exec(open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0])
# names W, Ee, S, N, G, E, U, a, ex, fc, B, rng, ctrl, tr are used by the
# helpers above: do not reuse them below
import io, time, zipfile, requests, hashlib
API = 'https://eeadmz1-downloads-api-appservice.azurewebsites.net'
META = ('https://discomap.eea.europa.eu/App/AQViewer/download?'
        'fqn=Airquality_Dissem.b2g.measurements&f=csv')
WD = f'{DIR}/eea_aq'; os.makedirs(WD, exist_ok=True)
saved = []
def save(df, name):
    p = f'{OUT}/{name}'; df.to_csv(p, index=False); saved.append(p)
ISO = {'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
       'Czechia': 'CZ', 'Czech Republic': 'CZ', 'Denmark': 'DK',
       'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR', 'Germany': 'DE',
       'Greece': 'GR', 'Hungary': 'HU', 'Ireland': 'IE', 'Italy': 'IT',
       'Latvia': 'LV', 'Lithuania': 'LT', 'Netherlands': 'NL', 'Poland': 'PL',
       'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI',
       'Spain': 'ES', 'Sweden': 'SE', 'United Kingdom': 'GB', 'Serbia': 'RS',
       'Norway': 'NO', 'Bosnia and Herzegovina': 'BA', 'Montenegro': 'ME',
       'North Macedonia': 'MK', 'Kosovo': 'XK', 'Turkey': 'TR',
       'Luxembourg': 'LU', 'Malta': 'MT', 'Cyprus': 'CY'}

# ---------- European coal sites, exits, placebo sites ----------
d5 = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv', low_memory=False)
d5['featureValue'] = pd.to_numeric(d5['featureValue'], errors='coerce')
d5['key'] = np.where(d5['featureType'] == 'LCPCharacteristics', d5['unit'],
                     d5['featureType'])
wv = d5.pivot_table(index=['LCPInspireId', 'reportingYear'], columns='key',
                    values='featureValue', aggfunc='sum').fillna(0)
for c_ in ['Coal', 'Lignite', 'SO2', 'NOX', 'DUST']:
    if c_ not in wv:
        wv[c_] = 0.0
fuels = [c_ for c_ in ['Coal', 'Lignite', 'Biomass', 'NaturalGas',
                       'LiquidFuels', 'Peat', 'OtherGases', 'OtherSolidFuels']
         if c_ in wv]
wv['solid'] = wv['Coal'] + wv['Lignite']
wv['share'] = wv['solid'] / wv[fuels].sum(axis=1).replace(0, np.nan)
info = d5.groupby('LCPInspireId').agg(lat=('Latitude', 'first'),
                                      lon=('Longitude', 'first'),
                                      country=('countryName', 'first'))
coal_ids = set(wv[(wv['share'] >= 0.5) & (wv['solid'] > 0)]
               .index.get_level_values(0))
COAL = info.loc[sorted(coal_ids)]
EX = pd.read_csv(f'{DIR}/eu_unit_exits.csv')
ev = (EX[EX['last_coal_year'].between(2017, 2022)]
      .sort_values('unit_mw_th', ascending=False)
      .drop_duplicates(['site', 'last_coal_year']))
ev = ev[(ev['site_so2_drop'] >= 1000) | (ev['site_nox_drop'] >= 1000)]
ev = ev.join(info[['lat', 'lon']], on='lcp').rename(
    columns={'last_coal_year': 't'})
ev['fid'] = 'EU' + ev['site'].astype(str)
print(f'European exit events: {len(ev)} (37 at filing); countries '
      f"{ev['country'].value_counts().to_dict()}")
burn = wv[wv['solid'] > 0].reset_index().groupby('LCPInspireId')[
    'reportingYear'].apply(set)
exited = set(EX['lcp'])
pl = info.loc[[l for l, ys in burn.items() if set(range(2016, 2023)) <= ys
               and l not in exited and l in info.index]]
pl = pl.round(2).drop_duplicates(['lat', 'lon']).copy()
pl['fid'] = ['EUP' + str(i) for i in range(len(pl))]
exl = info.loc[[l for l in exited if l in info.index]]
far_exit = [km(exl['lat'].values, exl['lon'].values, r.lat, r.lon).min() > 25
            for r in pl.itertuples()]
print(f'Never-exit coal sites (coal every year 2016-2022): {len(pl)}')

# ---------- station metadata ----------
p_m = f'{WD}/metadata.csv'
if not os.path.exists(p_m):
    r_ = requests.get(META, timeout=600); r_.raise_for_status()
    try:
        z = zipfile.ZipFile(io.BytesIO(r_.content))
        open(p_m, 'wb').write(z.read(z.namelist()[0]))
    except zipfile.BadZipFile:
        open(p_m, 'wb').write(r_.content)
M = pd.read_csv(p_m, low_memory=False)
def col(*keys):
    for c_ in M.columns:
        if all(k.lower() in c_.lower() for k in keys):
            return c_
    raise SystemExit(f'Metadata column {keys} not found. Columns: '
                     f'{list(M.columns)}. Send me this.')
c_sp, c_pol = col('sampling point', 'id'), col('pollutant')
c_lat, c_lon = col('latitude'), col('longitude')
c_typ = next((c_ for c_ in M.columns if 'station type' in c_.lower()), None)
c_cty = next((c_ for c_ in M.columns if c_.lower() in ('country',
              'country code', 'countrycode')), None)
print('Metadata columns used:', c_sp, '|', c_pol, '|', c_lat, '|', c_lon, '|',
      c_typ, '|', c_cty)
M = M.dropna(subset=[c_lat, c_lon]).copy()
import re as _re
def norm(v):        # Italian ids carry timestamps: ':' and ' ' become '_'
    return _re.sub(r'[:\s]', '_', str(v).split('/')[-1])
M['spid'] = M[c_sp].map(norm)
POLS = {'PM2.5': ['PM2.5'], 'SO2': ['SO2', 'Sulphur dioxide'],
        'NO2': ['NO2', 'Nitrogen dioxide']}
M['pol'] = None
for p_, names in POLS.items():
    pat = M[c_pol].astype(str)
    hit = pat.isin(names) | pat.str.endswith(
        {'PM2.5': '/6001', 'SO2': '/1', 'NO2': '/8'}[p_])
    M.loc[hit, 'pol'] = p_
M = M[M['pol'].notna()].drop_duplicates(['spid', 'pol'])
Sx_ = M[[c_lat, c_lon]].values.astype(float)
dcoal = np.full(len(M), np.inf)
for la, lo in COAL[['lat', 'lon']].values:
    dcoal = np.minimum(dcoal, km(Sx_[:, 0], Sx_[:, 1], la, lo))
M['dcoal'] = dcoal
dexit = np.full(len(M), np.inf)
for la, lo in exl[['lat', 'lon']].values:
    dexit = np.minimum(dexit, km(Sx_[:, 0], Sx_[:, 1], la, lo))
M['dexit'] = dexit
def ccode(row):
    if c_cty is not None:
        v = str(row[c_cty])
        if len(v) == 2 and v.isupper():
            return v
        if v in ISO:
            return ISO[v]
    raw = str(row[c_sp])
    if '/' in raw and len(raw.split('/')[0]) == 2:
        return raw.split('/')[0]
    import re
    m_ = re.search(r'\.([A-Z]{2})_', raw)
    return m_.group(1) if m_ else None
M['cc'] = M.apply(ccode, axis=1)
print('Country codes found:', M['cc'].value_counts().head(30).to_dict())
print(f"Sampling points: PM2.5 {(M.pol == 'PM2.5').sum()}, SO2 "
      f"{(M.pol == 'SO2').sum()}, NO2 {(M.pol == 'NO2').sum()}")

# near stations for events and placebo sites, background per country
NEAR = []
for grp, df in [('event', ev), ('placebo', pl[far_exit])]:
    for r in df.itertuples():
        d = km(Sx_[:, 0], Sx_[:, 1], r.lat, r.lon)
        for i in np.where(d <= 25)[0]:
            if grp == 'placebo' and M['dexit'].iloc[i] <= 25:
                continue
            NEAR.append(dict(grp=grp, fid=r.fid, spid=M['spid'].iloc[i],
                             pol=M['pol'].iloc[i], cc=M['cc'].iloc[i]))
NEAR = pd.DataFrame(NEAR)
ccs = sorted(set(NEAR['cc']))
rs = np.random.default_rng(20261113)
BG = []
for cc in ccs:
    for p_ in POLS:
        b_ = M[(M.cc == cc) & (M.pol == p_) & (M.dcoal > 50)]
        if c_typ is not None:
            bb = b_[b_[c_typ].astype(str).str.lower().str.contains('background')]
            b_ = bb if len(bb) >= 5 else b_
        if len(b_) > 40:
            b_ = b_.iloc[rs.choice(len(b_), 40, replace=False)]
        BG += [dict(cc=cc, pol=p_, spid=s_) for s_ in b_['spid']]
BG = pd.DataFrame(BG)
want = pd.concat([NEAR[['cc', 'pol', 'spid']], BG]).drop_duplicates()
print(f'Stations to download: {len(want)} sampling points in {len(ccs)} '
      f'countries ({len(NEAR)} near-station pairs, {len(BG)} background)')

# ---------- download (parquet), aggregate to station-month ----------
def urls(cc, pol, dataset, agg):
    body = dict(countries=[cc], cities=[], pollutants=[pol], dataset=dataset,
                source='API', aggregationType=agg,
                dateTimeStart='2016-01-01T00:00:00Z',
                dateTimeEnd='2024-01-01T00:00:00Z')
    for tries in range(4):
        try:
            r_ = requests.post(f'{API}/ParquetFile/urls', json=body, timeout=300)
            if r_.status_code == 200:
                lines = [l.strip() for l in r_.text.splitlines() if l.strip()]
                return [l for l in lines if l.startswith('http')]
            print('  status', r_.status_code, r_.text[:200])
        except Exception as e_:
            print('  retry', cc, pol, str(e_)[:80])
        time.sleep(10)
    return []
p_mm = f'{WD}/station_months.csv'
if not os.path.exists(p_mm):
    import pyarrow.parquet as pq
    out = []
    os.makedirs(f'{WD}/parts', exist_ok=True)
    for (cc, p_), g in want.groupby(['cc', 'pol']):
        p_part = f"{WD}/parts/{cc}_{p_.replace('.', '')}.csv"
        if os.path.exists(p_part):           # resume after a disconnect
            out.append(pd.read_csv(p_part)); continue
        n_before = len(out)
        ids = set(g['spid'])
        got = 0
        for ds in (2, 1):
            for agg in ('hour', 'day'):
                U_ = urls(cc, p_, ds, agg)
                sel = [u for u in U_ if any(s_ in _re.sub(r'%3A|:', '_', u) for s_ in ids)]
                if U_ and not sel:
                    print(f'  {cc} {p_} ds{ds} {agg}: {len(U_)} files, none '
                          f'matched by id; example {U_[0]}')
                for u in sel:
                    try:
                        t_ = pq.read_table(io.BytesIO(requests.get(
                            u, timeout=300).content)).to_pandas()
                    except Exception as e_:
                        print('  skip', u[-60:], str(e_)[:60]); continue
                    t_ = t_[(t_['Validity'] > 0) & (t_['Value'] >= 0)]
                    if t_.empty:
                        continue
                    t_['spid'] = t_['Samplingpoint'].map(norm)
                    t_['day'] = pd.to_datetime(t_['Start']).dt.strftime('%Y-%m-%d')
                    dd = t_.groupby(['spid', 'day'])['Value'].agg(['mean', 'size'])
                    if agg == 'hour':
                        dd = dd[dd['size'] >= 18]
                    dd = dd.reset_index()
                    dd['ym'] = dd['day'].str[:7].str.replace('-', '').astype(int)
                    mm = dd.groupby(['spid', 'ym'])['mean'].agg(['mean', 'size'])
                    mm = mm.reset_index().rename(columns={'mean': 'v',
                                                          'size': 'ndays'})
                    mm['pol'], mm['cc'], mm['ds'] = p_, cc, ds
                    out.append(mm); got += 1
        part = (pd.concat(out[n_before:]) if len(out) > n_before else
                pd.DataFrame(columns=['spid', 'ym', 'v', 'ndays', 'pol', 'cc',
                                      'ds']))
        part.to_csv(p_part, index=False)
        print(f'{cc} {p_}: {got} station files', flush=True)
    MM = pd.concat(out)
    MM = MM.sort_values('ds', ascending=False).drop_duplicates(
        ['spid', 'pol', 'ym'])          # verified (2) before up-to-date (1)
    MM.to_csv(p_mm, index=False)
MM = pd.read_csv(p_mm)
MM = MM[(MM['ndays'] >= 8) & (MM['v'] > 0)].copy()
MM['l'] = np.log(MM['v']); MM['yr'] = MM['ym'] // 100
bgset = set(zip(BG['spid'], BG['pol']))
MM['bg'] = [(s_, p_) in bgset for s_, p_ in zip(MM['spid'], MM['pol'])]
bgm = MM[MM['bg']].groupby(['cc', 'pol', 'ym'])['l'].mean().rename('bgl')
MM = MM.join(bgm, on=['cc', 'pol', 'ym']).dropna(subset=['bgl'])
MM['rel'] = MM['l'] - MM['bgl']
ann = MM.groupby(['spid', 'pol', 'yr']).agg(rel=('rel', 'mean'),
                                            n=('rel', 'size'), lv=('v', 'mean'))
ann = ann[ann['n'] >= 8]

def stat(fid, pol, y, grp):
    s_ = NEAR[(NEAR.fid == fid) & (NEAR.pol == pol) & (NEAR.grp == grp)]['spid']
    v, l = [], []
    for x in s_:
        if (x, pol, y - 1) in ann.index and (x, pol, y + 1) in ann.index:
            v.append(ann.at[(x, pol, y - 1), 'rel'] - ann.at[(x, pol, y + 1), 'rel'])
            l.append(ann.at[(x, pol, y - 1), 'lv'])
    return (np.mean(v), np.mean(l), len(v)) if v else (np.nan, np.nan, 0)

SUM = []
print('\n' + '=' * 70 + '\nEuropean causal arm (log points; fall positive)\n'
      + '=' * 70)
for p_ in POLS:
    E_ = []
    for r in ev.itertuples():
        dl, lv, n = stat(r.fid, p_, int(r.t), 'event')
        if n:
            E_.append(dict(fid=r.fid, name=r.name, country=r.country, t=r.t,
                           so2_drop=r.site_so2_drop, nox_drop=r.site_nox_drop,
                           dlog=dl, level=lv, n_st=n))
    P_ = []
    for r in pl[far_exit].itertuples():
        for y in range(2017, 2023):
            dl, lv, n = stat(r.fid, p_, y, 'placebo')
            if n:
                P_.append(dict(fid=r.fid, year=y, d=dl))
    E_, P_ = pd.DataFrame(E_), pd.DataFrame(P_)
    if len(E_) < 3 or len(P_) < 10:
        print(f'  {p_}: {len(E_)} events, {len(P_)} placebo changes; too few')
        SUM.append(dict(analysis=f'Europe {p_}', events=len(E_)))
        continue
    ym_ = P_.groupby('year')['d'].mean(); pool_ = P_['d'].mean()
    E_['adj'] = E_['t'].map(ym_).fillna(pool_)
    E_['fall'] = E_['dlog'] - E_['adj']
    sd_ = P_['d'].std()
    y_ = E_['fall'].values; m_ = y_.mean()
    se_ = sd_ / np.sqrt(len(E_))          # one event per site
    bs_ = cluster_boot(E_, lambda z: z['fall'].mean())
    p_pl = 2 * (1 - Phi(abs(m_ / se_)))
    p_wb = wild_p(y_, E_['fid'].values, 0.0)[1]
    p_bs = min(1.0, 2 * min((bs_ <= 0).mean(), (bs_ >= 0).mean()))
    print(f'  {p_}: {len(E_)} events ({E_.n_st.sum()} station pairs); '
          f'placebo {len(P_)} changes at {P_.fid.nunique()} sites (mean '
          f'{pool_:.4f}); fall {m_:.4f} log points, placebo 95% '
          f'[{m_ - 1.96 * se_:.4f}, {m_ + 1.96 * se_:.4f}], bootstrap '
          f'[{np.percentile(bs_, 2.5):.4f}, {np.percentile(bs_, 97.5):.4f}]; '
          f'p(0): placebo {p_pl:.3f}, wild {p_wb:.3f}, bootstrap {p_bs:.3f} '
          f'-> primary {max(p_pl, p_wb, p_bs):.3f}; unadjusted '
          f"{E_['dlog'].mean():.4f}; in ug/m3 about "
          f"{(E_['fall'] * E_['level']).mean():.3f}")
    SUM.append(dict(analysis=f'Europe {p_}', events=len(E_), mean=m_,
                    lo_placebo=m_ - 1.96 * se_, hi_placebo=m_ + 1.96 * se_,
                    lo_boot=np.percentile(bs_, 2.5),
                    hi_boot=np.percentile(bs_, 97.5),
                    p0_primary=max(p_pl, p_wb, p_bs),
                    unadjusted=E_['dlog'].mean(),
                    mean_ug=(E_['fall'] * E_['level']).mean()))
    save(E_, f'eu_causal_{p_.replace(".", "")}.csv')

# ---------- Europe Q2 rerun including Serbia ----------
print('\n' + '=' * 70 + '\nEurope Q2 valuation, with Serbia\n' + '=' * 70)
COST = {  # ETC/ATNI 2020/4 Table 27, EUR2019/t: NOx, PM10, SO2 (VOLY, VSL)
 'AT': (22542, 75653, 44979, 147529, 34630, 113123),
 'BE': (19464, 67390, 103329, 332491, 49322, 159275),
 'BG': (12010, 39745, 53333, 201070, 15114, 46368),
 'HR': (17497, 59756, 35252, 124874, 23315, 79013),
 'CZ': (15017, 49078, 57202, 183409, 22528, 71277),
 'DK': (7595, 25273, 25437, 80593, 17274, 54274),
 'EE': (2056, 6947, 5157, 17360, 2242, 6789),
 'FI': (4101, 14457, 13212, 42445, 5393, 17104),
 'FR': (19396, 63549, 42464, 135189, 34580, 111252),
 'DE': (20447, 74611, 49219, 173147, 34621, 116457),
 'GR': (7321, 26639, 27156, 94614, 11901, 36550),
 'HU': (20348, 68822, 50635, 169836, 24317, 77634),
 'IE': (10510, 29780, 12735, 32609, 26844, 77934),
 'IT': (28857, 108056, 107384, 384838, 26967, 93755),
 'LT': (5158, 18333, 11450, 40353, 7845, 25598),
 'LV': (4965, 19089, 17423, 64043, 8276, 28699),
 'NL': (23088, 79355, 61781, 191298, 42521, 135753),
 'PL': (8457, 26911, 27684, 83938, 14289, 42598),
 'PT': (8804, 31399, 43859, 151956, 10555, 35283),
 'RO': (15800, 52360, 42028, 141119, 20072, 61831),
 'RS': (7366, 22994, 37611, 120690, 15725, 48616),
 'SK': (16217, 50172, 49995, 151580, 19963, 60571),
 'SI': (24889, 84337, 72969, 242259, 28225, 93509),
 'ES': (12586, 42076, 41425, 130955, 22548, 71915),
 'SE': (5453, 18595, 10944, 34765, 6460, 20188),
 'GB': (16537, 55583, 56373, 174188, 37714, 117415),
}
EQ = EX[EX['last_coal_year'].between(2019, 2022)]
first_y = int(d5['reportingYear'].min())
rows, miss = [], set()
for x in EQ.itertuples():
    cc = ISO.get(x.country)
    if cc not in COST:
        miss.add(x.country); continue
    if x.lcp not in wv.index.get_level_values(0):
        continue
    g = wv.loc[x.lcp]
    yrs = [y for y in range(first_y, x.last_coal_year) if y in g.index]
    if x.last_coal_year - 1 not in g.index or not yrs:
        continue
    c = COST[cc]
    for basis, (cN, cP, cS) in [('VOLY', (c[0], c[2], c[4])),
                                ('VSL', (c[1], c[3], c[5]))]:
        dm = {y: g.at[y, 'SO2'] * cS + g.at[y, 'NOX'] * cN
              + g.at[y, 'DUST'] * cP for y in yrs}
        rows.append(dict(lcp=x.lcp, country=cc, basis=basis,
                         mw_th=x.unit_mw_th, dmg_peak=max(dm.values()),
                         dmg_final=dm[x.last_coal_year - 1]))
EV = pd.DataFrame(rows)
if miss:
    print('No damage cost for:', sorted(miss))
for basis in ['VOLY', 'VSL']:
    e_ = EV[(EV.basis == basis) & (EV.dmg_peak > 0)]
    big = e_[e_.mw_th >= 1000]
    sh = 1 - e_.dmg_final.sum() / e_.dmg_peak.sum()
    print(f'  {basis}: {len(e_)} unit exits; share before the last full year '
          f'{100 * sh:.1f}%; multiple {e_.dmg_peak.sum() / e_.dmg_final.sum():.2f}'
          f'; large units ({len(big)}) '
          f'{100 * (1 - big.dmg_final.sum() / big.dmg_peak.sum()):.1f}%')
    SUM.append(dict(analysis=f'Europe Q2 damages with Serbia ({basis})',
                    events=len(e_), mean=sh,
                    multiple=e_.dmg_peak.sum() / e_.dmg_final.sum()))
save(EV, 'eu_valuation_v2.csv')
save(pd.DataFrame(SUM), 'batchF_summary.csv')
print('\nSHA-256:')
for p in saved:
    print(f'  {os.path.basename(p)}  '
          f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
