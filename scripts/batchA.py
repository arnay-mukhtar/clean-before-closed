# POST-REGISTRATION, exploratory (descriptive Q1/Q2 extensions; no air-
# quality outcome data). Batch A, 10 Nov 2026. Four parts:
#  (A) Announcement. When did each exiting unit first appear in EIA-860
#      with a planned retirement year, and how much of its wind-down came
#      before that? Answers the objection that the wind-down was itself
#      caused by the decision to retire.
#  (B) Counterfactual robustness: matching that adds unit age, capacity,
#      heat rate and scrubber status; and a regression counterfactual
#      fitted on never-exit units and validated out of sample on later
#      exiters.
#  (C) Horizon and present value of the overstatement (1-10 years after
#      exit, 0/3/7% discounting); dollar values; benefit per MW retired.
#  (D) Which units wind down: shares by scrubber, age, size, heat rate.
# Conventions as in usunits.py / cfrobust.py: T = last year of coal
# burning, T-1 = last full year, "ten years before" = T-10.
# Downloads: EIA-860 2009-2024 generator files (eia.gov, cached to
# eia860_generator_panel.csv) and the EPA-EIA crosswalk (GitHub, cached
# to epa_eia_crosswalk.csv). Runs in about 15-30 minutes.
import os, io, re, zipfile, hashlib, requests
import numpy as np, pandas as pd
DIR = os.environ.get('CC_DIR', '/content/drive/MyDrive/coal_closures')
OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
ST2T = 0.90718474
rng = np.random.default_rng(20261110)
NB = int(os.environ.get('CC_NB', 2000))
COAL = {'BIT', 'SUB', 'LIG', 'RC', 'WC', 'ANT', 'SC', 'SGC'}
HDR = {'User-Agent': 'Mozilla/5.0 (research download)'}
# EPA central VSL, $7.4 million in 2006 dollars, CPI-U to 2024 dollars
# (annual averages 201.6 and 313.689); no income adjustment.
VSL = 7.4e6 * 313.689 / 201.6
saved = []
def save(df, name, **kw):
    p = f'{OUT}/{name}'; df.to_csv(p, **kw); saved.append(p)

def gid(x):
    s = str(x).strip().upper()
    return s[:-2] if re.fullmatch(r'\d+\.0', s) else s
def nu(uid):
    """normalised EPA unit key for the crosswalk"""
    f, u = str(uid).split('_', 1)
    return f + '_' + gid(u)

# ======================= data =======================
cols = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', nrows=0).columns
want = ['facilityid', 'unitid', 'year', 'statecode', 'so2mass', 'noxmass',
        'heatinput', 'grossload', 'primaryfuelinfo', 'so2controlinfo']
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=[c for c in want if c in cols])
for c in ['so2mass', 'noxmass', 'heatinput', 'grossload']:
    a[c] = pd.to_numeric(a.get(c, 0), errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
SCRUB = r'fgd|scrub|lime|alkali|sodium|magnesium|spray dry|dry sorbent'
a['scrub'] = a.get('so2controlinfo', pd.Series('', index=a.index)) \
    .fillna('').str.contains(SCRUB, case=False)
if 'so2controlinfo' in a:
    print('Most common SO2 control entries at coal units (scrubber flag '
          'in brackets):')
    vc = a.loc[a['coal'], 'so2controlinfo'].fillna('(none)').value_counts()
    for k, v in vc.head(8).items():
        print(f'   {v:6d}  {k} [{bool(re.search(SCRUB, k, re.I))}]')
else:
    print('No so2controlinfo column: scrubber status unavailable.')
coal_uids = set(a.loc[a['coal'], 'uid'])
A = a[a['uid'].isin(coal_uids)].pivot_table(
    index='uid', columns='year',
    values=['so2mass', 'noxmass', 'heatinput', 'grossload', 'scrub'],
    aggfunc={'so2mass': 'sum', 'noxmass': 'sum', 'heatinput': 'sum',
             'grossload': 'sum', 'scrub': 'max'}).fillna(0)
YEARS = sorted(A['heatinput'].columns)
Y0, YMAX = min(YEARS), max(YEARS)
for y in range(Y0, YMAX + 1):
    for v in ['so2mass', 'noxmass', 'heatinput', 'grossload', 'scrub']:
        if (v, y) not in A.columns:
            A[(v, y)] = 0.0
A = A.sort_index(axis=1)
IDX = list(A.index); POS = {u: i for i, u in enumerate(IDX)}
Hn, Sn, Nn = A['heatinput'].values, A['so2mass'].values, A['noxmass'].values
Gn, SCn = A['grossload'].values, A['scrub'].values.astype(float)
cy = lambda y: y - Y0
state = a.drop_duplicates('uid').set_index('uid')['statecode']
lastcoal = a[a['coal']].groupby('uid')['year'].max()
LC = lastcoal.reindex(IDX).fillna(-1).values
STn = state.reindex(IDX).values
fid_of = {u: int(u.split('_')[0]) for u in IDX}
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
U['fid'] = U['uid'].str.split('_').str[0].astype(int)
fcc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
cell_all = fcc.set_index('facilityid')['cell'].to_dict()
cell_all.update(U.groupby('fid')['cell'].first().to_dict())
def mm(cell, kind='MM'):
    c = int(cell) if pd.notna(cell) else -1
    if c < 0:
        return np.nan, np.nan
    return (mv[f'{kind}_SO2_high'].values[c] * ST2T,
            mv[f'{kind}_NOx_high'].values[c] * ST2T)

# ---------- EIA-860 generators and the EPA-EIA crosswalk ----------
def find_col(cs, *keys, avoid=()):
    for c in cs:
        lc = str(c).lower()
        if all(k in lc for k in keys) and not any(x in lc for x in avoid):
            return c
    return None
def read_sheet(raw):
    for i in range(min(8, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains('plant code|plant id|plant_code|plntcode').any():
            d = raw.iloc[i + 1:].copy()
            d.columns = raw.iloc[i].astype(str).str.strip()
            return d
    return None
p_gen = f'{DIR}/eia860_generator_panel.csv'
if not os.path.exists(p_gen):
    G = []
    for y in range(2009, 2025):
        content = None
        for base in ['https://www.eia.gov/electricity/data/eia860/xls/',
                     'https://www.eia.gov/electricity/data/eia860/archive/'
                     'xls/']:
            try:
                r = requests.get(f'{base}eia860{y}.zip', headers=HDR,
                                 timeout=300)
                if r.ok and r.content[:2] == b'PK':
                    content = r.content; break
            except Exception:
                pass
        if content is None:
            print(y, 'EIA-860 not found'); continue
        zz = zipfile.ZipFile(io.BytesIO(content))
        gf = [n for n in zz.namelist()
              if re.search(r'(^|/)(3_1_)?gen', n.lower())
              and n.lower().endswith(('.xlsx', '.xls'))
              and not re.search(r'wind|solar|multi|storage', n, re.I)]
        n_rows = 0
        for n in gf[:1]:
            book = pd.read_excel(io.BytesIO(zz.read(n)), sheet_name=None,
                                 header=None)
            for sh, raw in book.items():
                if re.search(r'propos|cancel', sh, re.I):
                    continue
                d = read_sheet(raw)
                if d is None:
                    continue
                c = list(d.columns)
                col = lambda *k, avoid=(): find_col(c, *k, avoid=avoid)
                get = lambda k: d[k] if k is not None else np.nan
                o = pd.DataFrame({
                    'file_year': y,
                    'sheet': 'retired' if re.search(r'retir', sh, re.I)
                    else 'operable',
                    'oris': pd.to_numeric(get(col('plant', 'code')
                                              or col('plant', 'id')),
                                          errors='coerce'),
                    'gen': get(col('generator', 'id') or col('gen', 'id')),
                    'status': get(col('status', avoid=('date',))),
                    'fuel': get(col('energy', 'source', '1')),
                    'mw': pd.to_numeric(get(col('nameplate')),
                                        errors='coerce'),
                    'op_year': pd.to_numeric(get(col('operating', 'year')),
                                             errors='coerce'),
                    'ret_year': pd.to_numeric(get(col('retire', 'year')),
                                              errors='coerce')})
                o = o.dropna(subset=['oris'])
                G.append(o); n_rows += len(o)
        print(f'{y}: {gf[:1]} -> {n_rows} generator rows', flush=True)
    G = pd.concat(G)
    G['gen'] = G['gen'].map(gid)
    G['status'] = G['status'].astype(str).str.strip().str.upper()
    G['fuel'] = G['fuel'].astype(str).str.strip().str.upper()
    G.to_csv(p_gen, index=False)
G = pd.read_csv(p_gen, dtype={'gen': str})
G['gen'] = G['gen'].map(gid); G['oris'] = G['oris'].astype(int)
p_cw = f'{DIR}/epa_eia_crosswalk.csv'
if not os.path.exists(p_cw):
    r = requests.get('https://raw.githubusercontent.com/USEPA/'
                     'camd-eia-crosswalk/master/epa_eia_crosswalk.csv',
                     headers=HDR, timeout=120)
    r.raise_for_status(); open(p_cw, 'wb').write(r.content)
CW = pd.read_csv(p_cw, dtype=str)
CW = CW.dropna(subset=['CAMD_PLANT_ID', 'CAMD_UNIT_ID', 'EIA_PLANT_ID',
                       'EIA_GENERATOR_ID'])
CW['uid'] = CW['CAMD_PLANT_ID'].astype(float).astype(int).astype(str) \
    + '_' + CW['CAMD_UNIT_ID'].map(gid)
CW['oris'] = CW['EIA_PLANT_ID'].astype(float).astype(int)
CW['gen'] = CW['EIA_GENERATOR_ID'].map(gid)
CW['mw'] = pd.to_numeric(CW['EIA_NAMEPLATE_CAPACITY'], errors='coerce')
CW = CW.drop_duplicates(['uid', 'oris', 'gen'])
# a generator shared by several EPA units is split equally among them
CW['mw_share'] = CW['mw'] / CW.groupby(['oris', 'gen'])['uid'] \
    .transform('nunique')
gens_of = CW.groupby('uid')[['oris', 'gen']].apply(
    lambda d: list(zip(d['oris'], d['gen']))).to_dict()
CAP = CW.groupby('uid')['mw_share'].sum(min_count=1)
opy = G.groupby(['oris', 'gen'])['op_year'].min()
coalgen = G[G['fuel'].isin(COAL)]
opy_plant = coalgen.groupby('oris')['op_year'].median()
OPY = {}
for u in IDX:
    gl = gens_of.get(nu(u), [])
    v = [opy.get(g) for g in gl if pd.notna(opy.get(g, np.nan))]
    OPY[u] = min(v) if v else opy_plant.get(fid_of[u], np.nan)
OPYn = np.array([OPY.get(u, np.nan) for u in IDX], dtype=float)
CAPn = CAP.reindex([nu(u) for u in IDX]).values.astype(float)
ex = set(U['uid'])
print(f'\nEIA-860 generator rows: {len(G):,} ({G.file_year.min()}-'
      f'{G.file_year.max()}); crosswalk matches {len({nu(u) for u in ex} & set(gens_of))} of '
      f'{len(ex)} exiting units; age known for '
      f'{np.isfinite([OPY.get(u, np.nan) for u in ex]).sum()}, capacity '
      f'for {CAP.reindex([nu(u) for u in ex]).notna().sum()}.')

def plant_boot(df, fn, nb=NB):
    """fn(sums_by_plant DataFrame) -> scalar; bootstrap by plant."""
    g = df.groupby('fid').sum(numeric_only=True)
    est = fn(g)
    idx = rng.integers(0, len(g), (nb, len(g)))
    bs = np.array([fn(g.iloc[i]) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi

# ======================= (A) announcement =======================
print('\n' + '=' * 70 + '\n(A) ANNOUNCEMENT: wind-down before and after a '
      'unit is first listed for retirement\n' + '=' * 70, flush=True)
P = G[(G['sheet'] == 'operable') & (G['ret_year'] > 1900)
      & (G['status'] != 'RE')]
first_list = P.groupby(['oris', 'gen'])['file_year'].min()
plan_at_first = P.sort_values('file_year').drop_duplicates(['oris', 'gen']) \
    .set_index(['oris', 'gen'])['ret_year']
YF = int(G['file_year'].min())
rows = []
for u in U.itertuples():
    T, i = int(u.last_coal_year), POS.get(u.uid)
    if i is None or T - 10 < Y0:
        continue
    gl = gens_of.get(nu(u.uid), [])
    if gl:
        v = [(first_list[g], plan_at_first[g]) for g in gl
             if g in first_list.index]
        method = 'crosswalk'
    else:
        q = P[(P['oris'] == u.fid) & P['fuel'].isin(COAL)
              & (P['ret_year'] <= T + 5)]
        v = list(zip(q['file_year'], q['ret_year']))
        method = 'plant'
    Ay, plan = (min(v)[0], min(v)[1]) if v else (np.nan, np.nan)
    rows.append(dict(uid=u.uid, fid=u.fid, T=T, A=Ay, planned=plan,
                     method=method, censored=Ay == YF,
                     s10=Sn[i, cy(T - 10)], s1=Sn[i, cy(T - 1)],
                     h10=Hn[i, cy(T - 10)], h1=Hn[i, cy(T - 1)],
                     sA=Sn[i, cy(int(np.clip(Ay - 1, T - 10, T - 1)))]
                     if np.isfinite(Ay) else Sn[i, cy(T - 1)],
                     hA=Hn[i, cy(int(np.clip(Ay - 1, T - 10, T - 1)))]
                     if np.isfinite(Ay) else Hn[i, cy(T - 1)]))
AN = pd.DataFrame(rows)
AN['lead'] = AN['T'] - AN['A']
listed = AN['A'].notna()
print(f'{len(AN)} exits with ten years of history. First listed with a '
      f'planned retirement year: {listed.sum()} '
      f'(crosswalk {(listed & (AN.method == "crosswalk")).sum()}, plant '
      f'fallback {(listed & (AN.method == "plant")).sum()}); never listed '
      f'before or after exit: {(~listed).sum()}.')
L = AN[listed]
print(f'  first listed in the first EIA-860 year available ({YF}; so '
      f'possibly earlier): {L.censored.sum()}')
print(f'  first listed after the last year of coal burning (A > T): '
      f'{(L.A > L["T"]).sum()}; years of notice T - A among the rest: '
      f'median {L.loc[L.A <= L["T"], "lead"].median():.0f}, '
      f'quartiles {L.loc[L.A <= L["T"], "lead"].quantile(.25):.0f}-'
      f'{L.loc[L.A <= L["T"], "lead"].quantile(.75):.0f}')
def decomp(d, base, pre, fin):
    k = d[base] > 0
    b = d.loc[k, base].sum()
    return (b - d.loc[k, pre].sum()) / b, \
        (d.loc[k, pre].sum() - d.loc[k, fin].sum()) / b, d.loc[k, fin].sum() / b
arow = []
groups = {'listed, not censored': AN[listed & ~AN.censored],
          'listed, incl. censored (upper bound on before-listing share)':
              AN[listed],
          'all exits (never listed = no notice)': AN}
for lab, d in groups.items():
    for var, base, pre, fin in [('SO2', 's10', 'sA', 's1'),
                                ('heat input', 'h10', 'hA', 'h1')]:
        e = decomp(d, base, pre, fin)
        _, lo, hi = plant_boot(d, lambda g: decomp(g, base, pre, fin)[0])
        arow.append(dict(group=lab, var=var, n=len(d), before_listing=e[0],
                         lo=lo, hi=hi, listing_to_last_full_year=e[1],
                         at_exit=e[2]))
        print(f'  {lab} ({len(d)} exits), {var}: of the cut from ten years '
              f'before, {100 * e[0]:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] '
              f'came before the first listing, {100 * e[1]:.0f}% between '
              f'listing and the last full year, {100 * e[2]:.0f}% at exit',
              flush=True)
# growth per year before vs after listing (units with >= 2 years notice)
E = L[(~L.censored) & (L.A <= L['T'] - 2) & (L.A - 4 >= Y0)].copy()
def gr(i, y0, y1, M, add):
    return (np.log(M[i, cy(y1)] + add) - np.log(M[i, cy(y0)] + add)) \
        / max(y1 - y0, 1)
grow = []
for r in E.itertuples():
    i, Ay, T = POS[r.uid], int(r.A), int(r.T)
    if Hn[i, cy(Ay - 4)] <= 0 or Hn[i, cy(T - 1)] <= 0:
        continue
    grow.append(dict(fid=r.fid, w=Hn[i, cy(Ay - 1)],
                     h_pre=gr(i, Ay - 4, Ay - 1, Hn, 0),
                     h_post=gr(i, Ay - 1, T - 1, Hn, 0),
                     s_pre=gr(i, Ay - 4, Ay - 1, Sn, 1),
                     s_post=gr(i, Ay - 1, T - 1, Sn, 1)))
GR = pd.DataFrame(grow)
if len(GR) >= 10:
    for v in ['h', 's']:
        GR[f'{v}_pre_w'] = GR[f'{v}_pre'] * GR['w']
        GR[f'{v}_post_w'] = GR[f'{v}_post'] * GR['w']
        f = lambda g, v=v: (g[f'{v}_post_w'].sum() - g[f'{v}_pre_w'].sum()) \
            / g['w'].sum()
        e, lo, hi = plant_boot(GR, f)
        pre = GR[f'{v}_pre_w'].sum() / GR['w'].sum()
        print(f'  {"heat input" if v == "h" else "SO2"}, {len(GR)} units with '
              f'2+ years of notice: {100 * pre:+.1f}% a year in the three '
              f'years before listing, {100 * (pre + e):+.1f}% a year from '
              f'listing to the last full year; change {100 * e:+.1f} points '
              f'[{100 * lo:+.1f}, {100 * hi:+.1f}] (log points, heat-weighted)')
save(AN, 'announce_units.csv', index=False)
save(pd.DataFrame(arow), 'announce_decomposition.csv', index=False)

# ======================= (B) counterfactual robustness =======================
print('\n' + '=' * 70 + '\n(B) COUNTERFACTUAL: richer matching and a '
      'regression counterfactual\n' + '=' * 70, flush=True)
def feats(rows_, T, names):
    h1, h5 = Hn[rows_, cy(T - 1)], Hn[rows_, cy(T - 5)]
    out = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for nm in names:
            if nm == 'trend': v = np.log(h1 / h5)
            elif nm == 'heat': v = np.log(h1)
            elif nm == 'so2rate': v = np.log((Sn[rows_, cy(T - 1)] + 1) / h1)
            elif nm == 'noxrate': v = np.log((Nn[rows_, cy(T - 1)] + 1) / h1)
            elif nm == 'age': v = T - OPYn[rows_]
            elif nm == 'cap': v = np.log(CAPn[rows_])
            elif nm == 'hrate':
                g = Gn[rows_, cy(T - 1)]
                v = np.where(g > 0, np.log(h1 / np.where(g > 0, g, 1)), np.nan)
            elif nm == 'scrub': v = SCn[rows_, cy(T - 1)]
            out.append(np.asarray(v, dtype=float))
    X = np.column_stack(out)
    X[~np.isfinite(X)] = np.nan
    return X
BASE = ['trend', 'heat', 'so2rate']
def donors(uid, T, names=BASE, k=5, horizon=5, exact_scrub=False):
    i = POS[uid]
    ok = (LC >= T + horizon) & (Hn[:, cy(T - 5):cy(T - 1) + 1] > 0).all(1)
    ok[i] = False
    pool = np.where(ok)[0]
    if len(pool) < 10 or Hn[i, cy(T - 1)] <= 0 or Hn[i, cy(T - 5)] <= 0:
        return None
    same = pool[STn[pool] == STn[i]]
    pool = same if len(same) >= k else pool
    if exact_scrub:
        s = pool[SCn[pool, cy(T - 1)] == SCn[i, cy(T - 1)]]
        pool = s if len(s) >= k else pool
    x0 = feats(np.array([i]), T, names)[0]
    X = feats(pool, T, names)
    keep = np.isfinite(x0) & np.isfinite(X).any(0)
    x0, X = x0[keep], X[:, keep]
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    sd = X.std(0) + 1e-9
    d = np.sqrt((((X - x0) / sd) ** 2).sum(1))
    return pool[np.argsort(d, kind='stable')[:k]]
def ratio_path(nn, T, H_=10):
    """donor totals at T+k over T-1, k = 1..H_ (SO2, NOx)."""
    sp, npre = Sn[nn, cy(T - 1)].sum(), Nn[nn, cy(T - 1)].sum()
    ks = range(1, H_ + 1)
    rs = np.array([Sn[nn, cy(T + k)].sum() / sp if sp > 0 else 1.0 for k in ks])
    rn = np.array([Nn[nn, cy(T + k)].sum() / npre if npre > 0 else 1.0
                   for k in ks])
    return rs, rn
def build(names=BASE, k=5, horizon=5, exact_scrub=False):
    rows_ = []
    for u in U.itertuples():
        T = int(u.last_coal_year)
        if u.uid not in POS or T - 5 < Y0 or T + horizon > YMAX:
            continue
        nn = donors(u.uid, T, names, k, horizon, exact_scrub)
        if nn is None:
            continue
        rs, rn = ratio_path(nn, T, horizon)
        ms, mn = mm(u.cell)
        if not np.isfinite(ms):
            continue
        rows_.append(dict(uid=u.uid, fid=u.fid, T=T, ms=ms, mn=mn,
                          d_final=u.so2_final * ms + u.nox_final * mn,
                          d_cf=u.so2_final * rs.mean() * ms
                          + u.nox_final * rn.mean() * mn,
                          d_t10=u.so2_t10 * ms + u.nox_t10 * mn,
                          rs=rs, rn=rn))
    return pd.DataFrame(rows_)
def validate(names, exact_scrub=False):
    rows_ = []
    for u in U['uid']:
        if u not in POS or u not in lastcoal.index:
            continue
        T = int(lastcoal[u]) - 6; i = POS[u]
        if T - 5 < Y0 or T + 5 > YMAX or Hn[i, cy(T - 1)] <= 0:
            continue
        nn = donors(u, T, names, exact_scrub=exact_scrub)
        if nn is None:
            continue
        rs, rn = ratio_path(nn, T, 5)
        ms, mn = mm(cell_all.get(fid_of[u], -1))
        if not np.isfinite(ms):
            continue
        sl = slice(cy(T + 1), cy(T + 5) + 1)
        rows_.append(dict(fid=fid_of[u],
                          act=Sn[i, sl].mean() * ms + Nn[i, sl].mean() * mn,
                          pred=Sn[i, cy(T - 1)] * rs.mean() * ms
                          + Nn[i, cy(T - 1)] * rn.mean() * mn))
    V = pd.DataFrame(rows_)
    return plant_boot(V, lambda g: g['act'].sum() / g['pred'].sum())
VARIANTS = [('as in the paper', BASE, False),
            ('+ age', BASE + ['age'], False),
            ('+ age, capacity', BASE + ['age', 'cap'], False),
            ('+ age, capacity, heat rate', BASE + ['age', 'cap', 'hrate'],
             False),
            ('+ age, capacity, heat rate, scrubber',
             BASE + ['age', 'cap', 'hrate', 'scrub'], False),
            ('all four, scrubber status matched exactly',
             BASE + ['age', 'cap', 'hrate'], True)]
brow, MAIN = [], None
for lab, names, exs in VARIANTS:
    C = build(names, exact_scrub=exs)
    e1 = plant_boot(C, lambda g: g.d_final.sum() / g.d_cf.sum())
    e2 = C.d_t10.sum() / C.d_cf.sum()
    vv = validate(names, exs)
    brow.append(dict(variant=lab, exits=len(C), last_full_year_x=e1[0],
                     lo=e1[1], hi=e1[2], ten_years_x=e2, validation=vv[0],
                     val_lo=vv[1], val_hi=vv[2]))
    print(f'  {lab}: {len(C)} exits; last-full-year {e1[0]:.2f}x '
          f'[{e1[1]:.2f}, {e1[2]:.2f}]; ten-years-before {e2:.2f}x; '
          f'validation actual/predicted {vv[0]:.2f} [{vv[1]:.2f}, {vv[2]:.2f}]',
          flush=True)
    if MAIN is None:
        MAIN = C
# balance: pre-exit heat-input trend, exiting unit vs donors
for lab, names in [('as in the paper', BASE),
                   ('all four', BASE + ['age', 'cap', 'hrate', 'scrub'])]:
    gaps = []
    for u in U.itertuples():
        T = int(u.last_coal_year)
        if u.uid not in POS or T - 5 < Y0 or T + 5 > YMAX:
            continue
        nn = donors(u.uid, T, names)
        if nn is None:
            continue
        i = POS[u.uid]
        gaps.append(dict(
            w=Hn[i, cy(T - 1)],
            e=Hn[i, cy(T - 1)] / Hn[i, cy(T - 5)],
            d=Hn[nn, cy(T - 1)].sum() / Hn[nn, cy(T - 5)].sum(),
            age_e=T - OPYn[i], age_d=np.nanmean(T - OPYn[nn]),
            sc_e=SCn[i, cy(T - 1)], sc_d=SCn[nn, cy(T - 1)].mean()))
    Bg = pd.DataFrame(gaps)
    print(f'  balance ({lab}): heat input T-1 over T-5, exiting '
          f'{np.average(Bg.e, weights=Bg.w):.2f} vs donors '
          f'{np.average(Bg.d, weights=Bg.w):.2f}; mean age '
          f'{Bg.age_e.mean():.0f} vs {Bg.age_d.mean():.0f} years; scrubbed '
          f'{100 * Bg.sc_e.mean():.0f}% vs {100 * Bg.sc_d.mean():.0f}%')
save(pd.DataFrame(brow), 'cf_matching_rich.csv', index=False)

# regression counterfactual: fitted on never-exit units only
RF = BASE + ['noxrate', 'age', 'cap', 'hrate', 'scrub']
never = np.where(LC >= YMAX - 1)[0]
later = np.array([POS[u] for u in U['uid'] if u in POS])
TY = list(range(Y0 + 5, YMAX - 4))
design = lambda rows_, T: feats(rows_, T, RF)
Xs, ys, yn, yr = [], [], [], []
for T in TY:
    r_ = never[(Hn[never, cy(T - 5):cy(T - 1) + 1] > 0).all(1)]
    if len(r_) == 0:
        continue
    sl = slice(cy(T + 1), cy(T + 5) + 1)
    Xs.append(design(r_, T))
    ys.append(np.log((Sn[r_, sl].mean(1) + 1) / (Sn[r_, cy(T - 1)] + 1)))
    yn.append(np.log((Nn[r_, sl].mean(1) + 1) / (Nn[r_, cy(T - 1)] + 1)))
    yr.append(np.full(len(r_), T))
Xs, ys, yn, yr = map(np.concatenate, (Xs, ys, yn, yr))
MED = np.nanmedian(Xs, 0)
def prep(X, years):
    X = np.where(np.isfinite(X), X, MED)
    D = (np.asarray(years)[:, None] == np.array(TY)[None, 1:]).astype(float)
    return np.column_stack([np.ones(len(X)), X, D])
Z = prep(Xs, yr)
bS = np.linalg.lstsq(Z, ys, rcond=None)[0]
bN = np.linalg.lstsq(Z, yn, rcond=None)[0]
smS = np.mean(np.exp(ys - Z @ bS)); smN = np.mean(np.exp(yn - Z @ bN))
print(f'  regression counterfactual: fitted on {len(ys):,} never-exit '
      f'unit-years ({len(never)} units); R2 SO2 '
      f'{1 - np.var(ys - Z @ bS) / np.var(ys):.2f}, NOx '
      f'{1 - np.var(yn - Z @ bN) / np.var(yn):.2f}')
def reg_ratio(i, T):
    Tc = int(np.clip(T, TY[0], TY[-1]))
    z = prep(feats(np.array([i]), T, RF), [Tc])
    return float(np.exp(z @ bS)[0] * smS), float(np.exp(z @ bN)[0] * smN)
vr = []
for u in U['uid']:
    if u not in POS or u not in lastcoal.index:
        continue
    i, T = POS[u], int(lastcoal[u]) - 6
    if T - 5 < Y0 or T + 5 > YMAX or Hn[i, cy(T - 1)] <= 0 \
            or Hn[i, cy(T - 5)] <= 0:
        continue
    rs, rn = reg_ratio(i, T)
    ms, mn = mm(cell_all.get(fid_of[u], -1))
    if not np.isfinite(ms):
        continue
    sl = slice(cy(T + 1), cy(T + 5) + 1)
    vr.append(dict(fid=fid_of[u],
                   act=Sn[i, sl].mean() * ms + Nn[i, sl].mean() * mn,
                   pred=Sn[i, cy(T - 1)] * rs * ms + Nn[i, cy(T - 1)] * rn * mn))
vv = plant_boot(pd.DataFrame(vr), lambda g: g['act'].sum() / g['pred'].sum())
rr = []
for u in U.itertuples():
    T = int(u.last_coal_year)
    if u.uid not in POS or T - 5 < Y0 or Hn[POS[u.uid], cy(T - 1)] <= 0 \
            or Hn[POS[u.uid], cy(T - 5)] <= 0:
        continue
    rs, rn = reg_ratio(POS[u.uid], T)
    ms, mn = mm(u.cell)
    if not np.isfinite(ms):
        continue
    rr.append(dict(uid=u.uid, fid=u.fid, T=T, rs=rs, rn=rn,
                   d_final=u.so2_final * ms + u.nox_final * mn,
                   d_cf=u.so2_final * rs * ms + u.nox_final * rn * mn,
                   d_t10=u.so2_t10 * ms + u.nox_t10 * mn))
RR = pd.DataFrame(rr)
e1 = plant_boot(RR, lambda g: g.d_final.sum() / g.d_cf.sum())
print(f'  regression counterfactual, {len(RR)} exits (no five-year '
      f'future needed): last-full-year {e1[0]:.2f}x [{e1[1]:.2f}, '
      f'{e1[2]:.2f}]; ten-years-before {RR.d_t10.sum() / RR.d_cf.sum():.2f}x;'
      f' out-of-sample validation on later exiters, actual/predicted '
      f'{vv[0]:.2f} [{vv[1]:.2f}, {vv[2]:.2f}]')
save(RR, 'cf_regression.csv', index=False)

# ======================= (C) horizon, PV, dollars, per MW =================
print('\n' + '=' * 70 + '\n(C) HORIZON, PRESENT VALUE, DOLLARS, PER MW\n'
      + '=' * 70, flush=True)
H10 = build(BASE, horizon=10)
print(f'  fixed sample: {len(H10)} exits with ten observed years of donor '
      f'futures (last coal year {H10["T"].min()}-{H10["T"].max()})')
crow = []
U_idx = U.set_index('uid')
so2f = U_idx.loc[H10.uid, 'so2_final'].values
noxf = U_idx.loc[H10.uid, 'nox_final'].values
RS = np.vstack(H10.rs.values); RN = np.vstack(H10.rn.values)
CFk = so2f[:, None] * RS * H10.ms.values[:, None] \
    + noxf[:, None] * RN * H10.mn.values[:, None]       # exits x 10 years
FIN, T10 = H10.d_final.values, H10.d_t10.values
for h in range(1, 11):
    own = build(BASE, horizon=h)
    m_fix = (FIN.sum() * h) / CFk[:, :h].sum()
    crow.append(dict(horizon=h, fixed_last=m_fix,
                     fixed_t10=(T10.sum() * h) / CFk[:, :h].sum(),
                     own_n=len(own),
                     own_last=own.d_final.sum() / own.d_cf.sum(),
                     own_t10=own.d_t10.sum() / own.d_cf.sum()))
    print(f'  horizon {h:2d} years: fixed sample last-full-year '
          f'{m_fix:.2f}x, ten-years-before {crow[-1]["fixed_t10"]:.2f}x | '
          f'all exits with {h} years observed ({len(own)}): '
          f'{crow[-1]["own_last"]:.2f}x, {crow[-1]["own_t10"]:.2f}x',
          flush=True)
yr_ratio = CFk.sum(0) / FIN.sum()
print('  counterfactual damages as a share of the last full year, years '
      '1-10 after exit:', ' '.join(f'{x:.2f}' for x in yr_ratio))
for rate in [0.0, 0.03, 0.07]:
    dfac = 1 / (1 + rate) ** np.arange(1, 11)
    pv_f, pv_c, pv_t = FIN.sum() * dfac.sum(), (CFk.sum(0) * dfac).sum(), \
        T10.sum() * dfac.sum()
    crow.append(dict(horizon=f'PV10 at {rate:.0%}', fixed_last=pv_f / pv_c,
                     fixed_t10=pv_t / pv_c))
    print(f'  present value over 10 years at {rate:.0%}: last-full-year '
          f'{pv_f / pv_c:.2f}x, ten-years-before {pv_t / pv_c:.2f}x')
save(pd.DataFrame(crow), 'cf_horizon.csv', index=False)

# dollars
k =(mv['MM_SO2_high'] > 0) & (mv['MD_SO2_high'] > 0)
imp = (mv.loc[k, 'MD_SO2_high'] / mv.loc[k, 'MM_SO2_high']).median()
print(f'\n  InMAP dollars per death implied by its own marginal values '
      f'(median over cells): ${imp / 1e6:.1f} million; dollars below use '
      f'EPA VSL ${VSL / 1e6:.1f} million (2024$)')
f, c, t = MAIN.d_final.sum(), MAIN.d_cf.sum(), MAIN.d_t10.sum()
drow = [dict(item=f'annual, {len(MAIN)} matched exits', counterfactual=c,
             last_full_year=f, ten_years_before=t)]
print(f'  annual, {len(MAIN)} matched exits: counterfactual {c:,.0f} deaths '
      f'(${c * VSL / 1e9:.0f} bn); last-full-year rule {f:,.0f} '
      f'(${f * VSL / 1e9:.0f} bn); ten-years-before rule {t:,.0f} '
      f'(${t * VSL / 1e9:.0f} bn). Overstatement ${(f - c) * VSL / 1e9:.0f}'
      f' bn and ${(t - c) * VSL / 1e9:.0f} bn a year.')
p_cum = f'{OUT}/cumulative_deaths.csv'
if os.path.exists(p_cum):
    cum = pd.read_csv(p_cum, index_col=0).iloc[:, 0]
    print('  cumulative through 2024 (from cumulative_deaths.csv):')
    for kk, v in cum.items():
        print(f'    {kk}: {v:,.0f} deaths, ${v * VSL / 1e9:,.0f} bn')
        drow.append(dict(item=f'cumulative: {kk}', deaths=v,
                         dollars_bn=v * VSL / 1e9))
# per MW retired
M = MAIN.assign(mw=CAP.reindex([nu(u) for u in MAIN.uid]).values)
Mk = M[M.mw > 0]
gw = Mk.mw.sum() / 1e3
print(f'  per GW retired ({len(Mk)} of {len(M)} matched exits with '
      f'capacity, {gw:.0f} GW): deaths avoided a year: counterfactual '
      f'{Mk.d_cf.sum() / gw:.1f}, last-full-year rule '
      f'{Mk.d_final.sum() / gw:.1f}, ten-years-before rule '
      f'{Mk.d_t10.sum() / gw:.1f}; in dollars per kW-year: '
      f'${Mk.d_cf.sum() * VSL / (Mk.mw.sum() * 1e3):.0f}, '
      f'${Mk.d_final.sum() * VSL / (Mk.mw.sum() * 1e3):.0f}, '
      f'${Mk.d_t10.sum() * VSL / (Mk.mw.sum() * 1e3):.0f}')
for lab, col in [('counterfactual', 'd_cf'), ('last full year', 'd_final'),
                 ('ten years before', 'd_t10')]:
    drow.append(dict(item=f'per GW-year: {lab}',
                     deaths=Mk[col].sum() / gw,
                     dollars_per_kw_year=Mk[col].sum() * VSL
                     / (Mk.mw.sum() * 1e3)))
save(pd.DataFrame(drow), 'valuation_dollars.csv', index=False)

# ======================= (D) which units wind down =======================
print('\n' + '=' * 70 + '\n(D) WHICH UNITS WIND DOWN (share of the cut '
      'from ten years before made by the last full year)\n' + '=' * 70,
      flush=True)
W = pd.read_csv(f'{DIR}/us_unit_exits.csv')
W['fid'] = W['uid'].str.split('_').str[0].astype(int)
W = W[W['uid'].isin(POS)].copy()
ii = W['uid'].map(POS).values
Tl = W['last_coal_year'].values.astype(int)
W['scrubbed'] = SCn[ii, Tl - 1 - Y0] > 0
W['age'] = Tl - OPYn[ii]
W['mw'] = CAPn[ii]
with np.errstate(divide='ignore', invalid='ignore'):
    W['hrate'] = np.where(Gn[ii, Tl - 1 - Y0] > 0,
                          Hn[ii, Tl - 1 - Y0] / Gn[ii, Tl - 1 - Y0], np.nan)
def terc(s, labs):
    q = s.quantile([1 / 3, 2 / 3]).values
    out = pd.Series(np.where(s <= q[0], labs[0],
                             np.where(s <= q[1], labs[1], labs[2])),
                    index=s.index, dtype=object)
    return out.where(s.notna())
W['age_g'] = terc(W['age'], ['youngest third', 'middle', 'oldest third'])
W['mw_g'] = terc(W['mw'], ['smallest third', 'middle', 'largest third'])
W['hr_g'] = terc(W['hrate'].where(W['hrate'].between(5, 30)),
                 ['most efficient third', 'middle', 'least efficient third'])
W['scr_g'] = np.where(W['scrubbed'], 'scrubbed', 'unscrubbed')
drow = []
for col, lab in [('scr_g', 'scrubber at T-1'), ('age_g', 'unit age'),
                 ('mw_g', 'capacity'), ('hr_g', 'heat rate')]:
    print(f'  by {lab}:')
    for gname, d in W[W[col].notna() & (W[col] != 'nan')].groupby(col):
        out = []
        for base, fin in [('so2_t10', 'so2_final'), ('heat_t10',
                                                      'heat_final')]:
            dd = d[d[base] > 0]
            e = plant_boot(dd, lambda g: 1 - g[fin].sum() / g[base].sum())
            out.append(e)
        drow.append(dict(split=lab, group=gname, units=len(d),
                         so2_share=out[0][0], so2_lo=out[0][1],
                         so2_hi=out[0][2], heat_share=out[1][0],
                         heat_lo=out[1][1], heat_hi=out[1][2]))
        print(f'    {gname} ({len(d)} units): SO2 {100 * out[0][0]:.0f}% '
              f'[{100 * out[0][1]:.0f}, {100 * out[0][2]:.0f}]; heat input '
              f'{100 * out[1][0]:.0f}% [{100 * out[1][1]:.0f}, '
              f'{100 * out[1][2]:.0f}]', flush=True)
save(pd.DataFrame(drow), 'winddown_heterogeneity.csv', index=False)

print('\nSHA-256 of the new outputs:')
for p in saved:
    print(f'  {os.path.basename(p)}  '
          f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
