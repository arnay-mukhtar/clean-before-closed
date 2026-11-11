# POST-REGISTRATION: registered SECONDARY analyses not yet reported (batch D,
# 10 Nov 2026). Each follows the registration's wording; where the wording
# leaves a choice open, the choice is stated in the printout.
#  (A) Downwind design, daily (registration 5, Q3 item 4): daily AQS PM2.5
#      at monitors within 25 km of the primary events, daily ERA5-Land wind;
#      (downwind minus upwind) log PM2.5, before minus after; tested against
#      zero and against 1.1 ug/m3. The frozen placebo draws give the SE.
#  (B) US recent closures (68 clean full-plant closures dated monthly):
#      12 months before vs 12 months after the closure month, monitors
#      within 25 km, relative to state background, against the frozen
#      predictions in model_predictions_us.csv.
#  (C) Full-field prediction (Q3 item 5): predicted change from every coal
#      facility's emissions change between the two years, not just the
#      event plant, compared with the observed change; the same at the
#      never-exit placebo plants (explains the control adjustment).
#  (D) Europe Q2: valuation of European unit exits with EEA country damage
#      costs per tonne (ETC/ATNI 2020/4, Table 27, EUR2019, VOLY and VSL).
DIR = '/content/drive/MyDrive/coal_closures'
exec(open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0])
# names W, Ee, S, N, G, E, U, a, ex, fc, B, rng, ctrl, tr are used by the
# helpers above: do not reuse them below
import io, time, zipfile, requests, hashlib
saved = []
def save(df, name):
    p = f'{OUT}/{name}'; df.to_csv(p, index=False); saved.append(p)
TPY = 907184.74e6 / (365.25 * 86400)
ST2T = 0.90718474

# frozen inputs used here must match the pre-registration deposit
fpr2 = pd.read_csv(f'{DIR}/fingerprints.csv').set_index('file')['sha256']
for f in ['downwind_placebo_draws.csv', 'wind_daily_plants.csv',
          'model_predictions_us.csv', 'us_closure_dates.csv',
          'model_predictions_us_events_v2.csv']:
    h = hashlib.sha256(open(f'{DIR}/{f}', 'rb').read()).hexdigest()
    print(f'  {f}: {"matches" if h == fpr2.get(f) else "DIFFERS FROM"} '
          'the registered fingerprint')

def bearing(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = np.sin(o2 - o1) * np.cos(a2)
    y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(o2 - o1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def wclust_se(v, groups):
    """Placebo-style SE of a mean when each event's value is v_e times a
    common SD, events at the same plant perfectly correlated."""
    s = pd.Series(np.asarray(v, float)).groupby(np.asarray(groups)).sum()
    return np.sqrt((s ** 2).sum()) / len(v)

def report(lab, d, ycol, sd_unit, scale_col=None, alt=None):
    """Registered inference trio for a mean: placebo SE, cluster bootstrap
    by plant, wild cluster bootstrap by plant. Most conservative p first."""
    y = d[ycol].values; m = y.mean()
    sc = d[scale_col].values if scale_col else np.ones(len(d))
    se = sd_unit * wclust_se(sc, d['fid'].values)
    bs = cluster_boot(d, lambda z: z[ycol].mean())
    p_pl = 2 * (1 - Phi(abs(m / se)))
    p_wb = wild_p(y, d['fid'].values, 0.0)[1]
    p_bs = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    r = dict(analysis=lab, events=len(d), plants=d['fid'].nunique(),
             mean=m, se_placebo=se, lo_placebo=m - 1.96 * se,
             hi_placebo=m + 1.96 * se, lo_boot=np.percentile(bs, 2.5),
             hi_boot=np.percentile(bs, 97.5), p0_placebo=p_pl,
             p0_wild=p_wb, p0_boot=p_bs, p0_primary=max(p_pl, p_wb, p_bs))
    if alt is not None:
        q_pl = Phi((m - alt) / se)
        q_wb = wild_p(y, d['fid'].values, alt, True)[1]
        q_bs = (bs >= alt).mean()
        r.update(p_below_alt_placebo=q_pl, p_below_alt_wild=q_wb,
                 p_below_alt_boot=q_bs, p_below_alt_primary=max(q_pl, q_wb, q_bs))
    print(f"  {lab}: {len(d)} events at {d['fid'].nunique()} plants; mean "
          f"{m:.3f}; placebo 95% [{m - 1.96 * se:.3f}, {m + 1.96 * se:.3f}]; "
          f"bootstrap [{r['lo_boot']:.3f}, {r['hi_boot']:.3f}]; p(0): placebo "
          f"{p_pl:.3f}, wild {p_wb:.3f}, bootstrap {p_bs:.3f} -> primary "
          f"{r['p0_primary']:.3f}"
          + (f"; p(mean >= {alt}) primary {r['p_below_alt_primary']:.4f}"
             if alt is not None else ''))
    return r

SUM = []
P2 = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
F = pd.read_csv(f'{OUT}/events_pm25_35km.csv')          # 51 primary events
lay = P2.drop_duplicates('fid').set_index('fid')['layer']

# ======================================================================
# (A) DOWNWIND, DAILY
# ======================================================================
print('\n' + '=' * 70 + '\n(A) Downwind design, daily PM2.5\n' + '=' * 70)
mon = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv',
                             usecols=['site', 'lat', 'lon'])
                 for y in range(2000, 2026)]).drop_duplicates('site')
Mx = mon[['lat', 'lon']].values
pairs_e = []
for r in F.itertuples():
    d = km(Mx[:, 0], Mx[:, 1], r.lat, r.lon)
    for i in np.where(d <= 25)[0]:
        pairs_e.append(dict(fid=r.fid, event_year=r.event_year,
                            site=mon['site'].iloc[i],
                            bearing=bearing(r.lat, r.lon, Mx[i, 0], Mx[i, 1])))
pairs_e = pd.DataFrame(pairs_e)
need = {}
for r in pairs_e.itertuples():
    for y in (r.event_year - 1, r.event_year + 1):
        need.setdefault(y, set()).add(r.site)
p_d = f'{DIR}/pm25_daily_events.csv'
if not os.path.exists(p_d):
    parts = []
    for y in sorted(need):
        for tries in range(4):
            try:
                rq = requests.get(f'https://aqs.epa.gov/aqsweb/airdata/'
                                  f'daily_88101_{y}.zip', timeout=900)
                rq.raise_for_status(); break
            except Exception as e_:
                print('  retry', y, str(e_)[:60]); time.sleep(20)
        z = zipfile.ZipFile(io.BytesIO(rq.content))
        d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False,
                        usecols=['State Code', 'County Code', 'Site Num',
                                 'Date Local', 'Arithmetic Mean'])
        d['site'] = (d['State Code'].astype(str) + '-' + d['County Code']
                     .astype(str) + '-' + d['Site Num'].astype(str))
        d = d[d['site'].isin(need[y])]
        parts.append(d.groupby(['site', 'Date Local'])['Arithmetic Mean']
                     .mean().reset_index())
        print(y, 'daily rows kept', len(parts[-1]), flush=True)
    pd.concat(parts).rename(columns={'Date Local': 'date',
                                     'Arithmetic Mean': 'pm'}) \
        .to_csv(p_d, index=False)
pmd = pd.read_csv(p_d)
pmd = pmd[pmd['pm'] > 0].copy()
pmd['l'] = np.log(pmd['pm']); pmd['yr'] = pmd['date'].str[:4].astype(int)
wd = pd.read_csv(f'{DIR}/wind_daily_plants.csv')
wd['to'] = (np.degrees(np.arctan2(wd['u'], wd['v'])) + 360) % 360

def du_table(pairs, pm, key):
    X = pairs.merge(pm, on='site').merge(
        wd[['facilityid', 'date', 'to']].rename(columns={'facilityid': key}),
        on=[key, 'date'])
    diff = np.abs((X['to'] - X['bearing'] + 180) % 360 - 180)
    X['dir'] = np.where(diff <= 45, 'down', np.where(diff >= 135, 'up',
                                                     'side'))
    g = X[X['dir'] != 'side'].groupby([key, 'site', 'yr', 'dir'])
    s = g['l'].agg(['mean', 'size']).unstack('dir')
    s = s[(s[('size', 'down')] >= 10) & (s[('size', 'up')] >= 10)]
    out = (s[('mean', 'down')] - s[('mean', 'up')]).rename('du').reset_index()
    lv = X.groupby(['site', 'yr'])['pm'].mean()
    return out, lv, (X['dir'] == 'down').mean()

DU, LVd, share_e = du_table(pairs_e, pmd, 'fid')
DU = DU.set_index(['fid', 'site', 'yr'])['du']
ev = []
for r in F.itertuples():
    b, a_ = r.event_year - 1, r.event_year + 1
    v, l = [], []
    for s_ in pairs_e.loc[(pairs_e.fid == r.fid)
                          & (pairs_e.event_year == r.event_year), 'site']:
        if (r.fid, s_, b) in DU.index and (r.fid, s_, a_) in DU.index:
            v.append(DU[(r.fid, s_, b)] - DU[(r.fid, s_, a_)])
            l.append(LVd[(s_, b)])
    if v:
        ev.append(dict(fid=r.fid, name=r.name, event_year=r.event_year,
                       n_mon=len(v), dlog=np.mean(v), level=np.mean(l),
                       pred_main=r.pred_main))
DW = pd.DataFrame(ev)
print(f'Events with a qualifying monitor (10+ downwind and 10+ upwind days '
      f'in both years): {len(DW)} (35 expected at filing); share of '
      f'downwind days {share_e:.2f}')
# control adjustment: the same statistic at never-exit plants, by year
pc = pd.read_csv(f'{DIR}/pm25_daily_controls.csv')
pc = pc[pc['pm'] > 0].copy()
pc['l'] = np.log(pc['pm']); pc['yr'] = pc['date'].str[:4].astype(int)
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
DUc, _, _ = du_table(pairs_c, pc, 'facilityid')
Dc = {}
for (f, si), gg in DUc.groupby(['facilityid', 'site']):
    gg = gg.set_index('yr')['du']
    for y in range(2004, 2019):
        if y - 1 in gg.index and y + 1 in gg.index:
            Dc.setdefault((f, y), []).append(gg[y - 1] - gg[y + 1])
Dc = pd.Series({k: np.mean(v) for k, v in Dc.items()})
ym_dw = Dc.groupby(level=1).mean(); pool_dw = Dc.mean()
DW['adj'] = DW['event_year'].map(ym_dw).fillna(pool_dw)
DW['fall_raw'] = DW['dlog'] * DW['level']
DW['fall'] = (DW['dlog'] - DW['adj']) * DW['level']
sd_dw = pd.read_csv(f'{DIR}/downwind_placebo_draws.csv')['d'].std()
print(f'Placebo: {len(Dc)} plant-years at {Dc.index.get_level_values(0).nunique()}'
      f' never-exit plants, pooled mean {pool_dw:.4f} log; frozen draws SD '
      f'{sd_dw:.4f} log')
r = report('Downwind daily, control-adjusted (ug/m3)', DW, 'fall', sd_dw,
           'level', 1.1); SUM.append(r)
r = report('Downwind daily, unadjusted (ug/m3)', DW, 'fall_raw', sd_dw,
           'level'); SUM.append(r)
print(f'  rough downwind-day prediction (mean annual prediction / share of '
      f'downwind days): {DW.pred_main.mean() / share_e:.2f} ug/m3')
save(DW, 'downwind_daily_events.csv')

# ======================================================================
# (B) 68 RECENT CLOSURES, MONTHLY DATING
# ======================================================================
print('\n' + '=' * 70 + '\n(B) US recent closures, 12 months either side\n'
      + '=' * 70)
print('Reading AQS PM2.5 ...', flush=True)
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
PM = prep(q, 'pm', 8)
qm = q[(q['ndays'] >= 8) & (q['pm'] > 0)].copy(); del q
qm['l'] = np.log(qm['pm'])
bgs = set(PM['st'].index[PM['st']['bg']])
bgm = qm[qm['site'].isin(bgs)].groupby(['state', 'ym'])['l'].mean().rename('bgl')
qm = qm.join(bgm, on=['state', 'ym']).dropna(subset=['bgl'])
qm['rel'] = qm['l'] - qm['bgl']
MR = qm.set_index(['site', 'ym'])[['rel', 'pm']].sort_index()
site_months = qm.groupby('site')['ym'].apply(set)
stA, SxA = PM['st'], PM['Sx']

def window(e, sign):
    p = pd.Period(str(int(e)), 'M')
    return [int((p + sign * k).strftime('%Y%m')) for k in range(1, 13)]

def mchange(lat, lon, e, radius, allowed=None):
    pre, post = set(window(e, -1)), set(window(e, 1))
    d = km(SxA[:, 0], SxA[:, 1], lat, lon)
    v, l = [], []
    for s_ in stA.index[d <= radius]:
        if allowed is not None and s_ not in allowed:
            continue
        ms = site_months.get(s_, set())
        a1, a2 = sorted(ms & pre), sorted(ms & post)
        if len(a1) < 8 or len(a2) < 8:
            continue
        v.append(MR.loc[[(s_, m) for m in a1], 'rel'].mean()
                 - MR.loc[[(s_, m) for m in a2], 'rel'].mean())
        l.append(MR.loc[[(s_, m) for m in a1], 'pm'].mean())
    if not v:
        return np.nan, np.nan, 0
    return float(np.mean(v)), float(np.mean(l)), len(v)

pr = pd.read_csv(f'{DIR}/model_predictions_us.csv')
pl_ = pd.read_csv(f'{DIR}/plants.csv')
CL = pr.merge(pl_[['plant_id', 'lat', 'lon']], on='plant_id')
CL['fid'] = CL['plant_id'].astype(str).str.replace('EIA', '').astype(float) \
              .astype(int)
CL['layer'] = CL['fid'].map(lay).fillna(2).astype(int)
CL['pred'] = np.where(CL['layer'] == 1, CL['contrast_ug_low'],
                      CL['contrast_ug_high'])
obs = [mchange(r.lat, r.lon, r.event_ym, 25) for r in CL.itertuples()]
CL['dlog'], CL['level'], CL['n_obs'] = zip(*obs)
print(f'{len(CL)} closures; with a qualifying monitor within 25 km: '
      f'{int((CL.n_obs > 0).sum())} (31 expected at filing); last closure '
      f'month {CL.event_ym.max()}')
# placebo: never-exit (operating) plants, same months, monitors > 25 km
# from every exit plant and every closure
ctl = pd.read_csv(f'{DIR}/us_operating_controls.csv')
far = np.full(len(SxA), np.inf)
for la, lo in np.vstack([ex[['latitude', 'longitude']].values,
                         CL[['lat', 'lon']].values]):
    far = np.minimum(far, km(SxA[:, 0], SxA[:, 1], la, lo))
allowed = set(stA.index[far > 25])
months = sorted(CL.loc[CL.n_obs > 0, 'event_ym'].unique())
PC = []
for c in ctl.itertuples():
    for m in months:
        dl, lv, n = mchange(c.lat, c.lon, m, 25, allowed)
        if n:
            PC.append((c.Index, m, dl, lv))
PC = pd.DataFrame(PC, columns=['ctrl', 'ym', 'd', 'lv'])
print(f'Placebo: {len(PC)} plant-months at {PC.ctrl.nunique()} operating '
      f'plants; pooled mean {PC.d.mean():.4f} log')
ym_c = PC.groupby('ym')['d'].mean(); pool_c = PC['d'].mean()
C2 = CL[CL.n_obs > 0].copy()
C2['adj'] = C2['event_ym'].map(ym_c).fillna(pool_c)
C2['fall_raw'] = C2['dlog'] * C2['level']
C2['fall'] = (C2['dlog'] - C2['adj']) * C2['level']
sd_c = PC['d'].std()                 # log points; scaled by each level
r = report('68 closures, control-adjusted (ug/m3)', C2, 'fall', sd_c,
           'level', 1.1); SUM.append(r)
r = report('68 closures, unadjusted (ug/m3)', C2, 'fall_raw', sd_c,
           'level'); SUM.append(r)
C3 = C2[C2['pred'].notna()]
b = slope0(C3['pred'], C3['fall'])
bs = cluster_boot(C3, lambda z: slope0(z['pred'], z['fall']))
print(f"  mean frozen prediction {C3['pred'].mean():.3f} ug/m3 (SO2 and NOx "
      f"only); slope of observed on predicted {b:.2f} "
      f"[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}], "
      f"p(slope = 1) {boot_slope_p(bs):.4f}")
SUM.append(dict(analysis='68 closures, slope on prediction', events=len(C3),
                mean=b, lo_boot=np.percentile(bs, 2.5),
                hi_boot=np.percentile(bs, 97.5), p_slope1=boot_slope_p(bs),
                mean_prediction=C3['pred'].mean()))
save(C2, 'closures68_events.csv')

# ======================================================================
# (C) FULL-FIELD PREDICTION
# ======================================================================
print('\n' + '=' * 70 + '\n(C) Full-field prediction: every coal facility\n'
      + '=' * 70)
Rr = np.load(f'{DIR}/isrm_rows_allcoal.npz')
rc = {c: i for i, c in enumerate(Rr['cells'])}
nr = len(rc)
RS = {L: Rr[f'pSO4_L{L}'] for L in (1, 2)}
RN = {L: Rr[f'pNO3_L{L}'] for L in (1, 2)}
a2 = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                 usecols=['facilityid', 'year', 'so2mass', 'noxmass'])
for c_ in ['so2mass', 'noxmass']:
    a2[c_] = pd.to_numeric(a2[c_], errors='coerce').fillna(0)
fy = a2.groupby(['facilityid', 'year'])[['so2mass', 'noxmass']].sum()
nat = a2.groupby('year')[['so2mass', 'noxmass']].sum()
fcc = fc[fc['cell'].isin(rc)].set_index('facilityid')['cell']
flay = pd.Series(2, index=fcc.index); flay.update(lay.reindex(fcc.index).dropna())
fields = {}
def field(y, layer_rule):
    k = (y, layer_rule)
    if k in fields:
        return fields[k]
    dS = {1: np.zeros(nr, 'f4'), 2: np.zeros(nr, 'f4')}
    dN = {1: np.zeros(nr, 'f4'), 2: np.zeros(nr, 'f4')}
    for f, cell in fcc.items():
        e0 = fy.loc[(f, y - 1)] if (f, y - 1) in fy.index else None
        e1 = fy.loc[(f, y + 1)] if (f, y + 1) in fy.index else None
        s0 = e0['so2mass'] if e0 is not None else 0.0
        n0 = e0['noxmass'] if e0 is not None else 0.0
        s1 = e1['so2mass'] if e1 is not None else 0.0
        n1 = e1['noxmass'] if e1 is not None else 0.0
        L = flay[f] if layer_rule == 'main' else (2 if layer_rule == 'high'
                                                   else 1)
        dS[L][rc[cell]] += s0 - s1; dN[L][rc[cell]] += n0 - n1
    v = sum(dS[L] @ RS[L] + dN[L] @ RN[L] for L in (1, 2)) * TPY
    fields[k] = v.astype('f8')
    return fields[k]

def pairs_for(lat, lon, y, radius, allowed_set=None):
    st, Sx, yrs = PM['st'], PM['Sx'], PM['yrs']
    b_, a_ = y - 1, y + 1
    d = km(Sx[:, 0], Sx[:, 1], lat, lon)
    out = []
    for s_ in st.index[d <= radius]:
        if allowed_set is not None and s_ not in allowed_set:
            continue
        if not {b_, a_} <= yrs.get(s_, set()) or st.at[s_, 'cell'] < 0:
            continue
        bg = [x for x in st.index[(st['state'] == st.at[s_, 'state'])
                                  & st['bg'] & (st['cell'] >= 0)]
              if {b_, a_} <= yrs.get(x, set())]
        if bg:
            out.append((int(st.at[s_, 'cell']),
                        st.loc[bg, 'cell'].values.astype(int)))
    return out

def contrast(v, prs):
    return np.mean([v[mc] - v[bc].mean() for mc, bc in prs]) if prs else np.nan

inc = fy[fy.index.get_level_values(0).isin(fcc.index)].groupby(level=1).sum()
cov = []
for y in sorted(set(F['event_year']) | set(range(2005, 2019))):
    if y + 1 > nat.index.max():
        continue
    tot = (nat.loc[y - 1] - nat.loc[y + 1])
    sub = inc
    cov.append((y, (sub.loc[y - 1, 'so2mass'] - sub.loc[y + 1, 'so2mass'])
                / tot['so2mass'],
                (sub.loc[y - 1, 'noxmass'] - sub.loc[y + 1, 'noxmass'])
                / tot['noxmass']))
cov = pd.DataFrame(cov, columns=['year', 'so2_share', 'nox_share'])
print('Share of the national CAMPD change (y-1 to y+1) from facilities with '
      f'an ISRM row: SO2 median {cov.so2_share.median():.2f}, NOx median '
      f'{cov.nox_share.median():.2f}')
FF = []
for r in F.itertuples():
    prs = pairs_for(r.lat, r.lon, int(r.event_year), 35)
    o = dict(fid=r.fid, name=r.name, event_year=r.event_year,
             fall=r.fall, fall_raw=r.fall_raw, pred_main=r.pred_main,
             ppm_main=r.ppm_main, n_pairs=len(prs))
    for rule in ['main', 'high', 'low']:
        o[f'ff_{rule}'] = contrast(field(int(r.event_year), rule), prs) \
            + r.ppm_main
    FF.append(o)
FF = pd.DataFrame(FF)
print(f'{len(FF)} events; mean event-plant prediction {FF.pred_main.mean():.3f}'
      f'; mean full-field prediction {FF.ff_main.mean():.3f} (high layer '
      f'{FF.ff_high.mean():.3f}, low {FF.ff_low.mean():.3f}) ug/m3')
print(f'  observed: control-adjusted {FF.fall.mean():.3f}, unadjusted '
      f'{FF.fall_raw.mean():.3f}')
for ycol in ['fall_raw', 'fall']:
    b = slope0(FF['ff_main'], FF[ycol])
    bs = cluster_boot(FF, lambda z: slope0(z['ff_main'], z[ycol]))
    rk = FF[['ff_main', ycol]].corr('spearman').iloc[0, 1]
    print(f'  {ycol} on full-field prediction: slope {b:.2f} '
          f'[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}], '
          f'p(slope = 1) {boot_slope_p(bs):.4f}; rank correlation {rk:.2f}')
    SUM.append(dict(analysis=f'full field: {ycol} on prediction',
                    events=len(FF), mean=b, lo_boot=np.percentile(bs, 2.5),
                    hi_boot=np.percentile(bs, 97.5), p_slope1=boot_slope_p(bs),
                    mean_prediction=FF.ff_main.mean(), rank_corr=rk))
d_ = FF['fall_raw'] - FF['ff_main']
se_ = np.sqrt((pd.Series(d_.values - d_.mean()).groupby(
    FF['fid'].values).sum() ** 2).sum()) / len(d_)
print(f'  unadjusted observed minus full-field prediction: {d_.mean():.3f} '
      f'(cluster SE {se_:.3f})')
save(FF, 'fullfield_events.csv')
# placebo plants: does the model predict their improvement?
st0 = PM['st']; allowed35 = set(st0.index[st0['d_exit'] > 35])
annP = PM['ann']
PP = []
for c in ctrl.itertuples():
    for y in range(2005, 2019):
        prs = pairs_for(c.latitude, c.longitude, y, 35, allowed35)
        if not prs:
            continue
        d = km(PM['Sx'][:, 0], PM['Sx'][:, 1], c.latitude, c.longitude)
        v, l = [], []
        for s_ in st0.index[d <= 35]:
            if s_ in allowed35 and (s_, y - 1) in annP.index \
                    and (s_, y + 1) in annP.index:
                v.append(annP.at[(s_, y - 1), 'rel'] - annP.at[(s_, y + 1), 'rel'])
                l.append(annP.at[(s_, y - 1), 'lv'])
        if v:
            PP.append(dict(fid=c.facilityid, year=y, obs=np.mean(v) * np.mean(l),
                           ff_main=contrast(field(y, 'main'), prs)))
PP = pd.DataFrame(PP)
b = slope0(PP['ff_main'], PP['obs'])
bs = cluster_boot(PP, lambda z: slope0(z['ff_main'], z['obs']))
print(f'Placebo plants: {len(PP)} plant-years at {PP.fid.nunique()} plants; '
      f'observed improvement {PP.obs.mean():.3f}, full-field prediction '
      f'{PP.ff_main.mean():.3f} ug/m3; slope {b:.2f} '
      f'[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]')
SUM.append(dict(analysis='full field at placebo plants', events=len(PP),
                plants=PP.fid.nunique(), mean=PP.obs.mean(),
                mean_prediction=PP.ff_main.mean(), slope=b,
                lo_boot=np.percentile(bs, 2.5), hi_boot=np.percentile(bs, 97.5)))
save(PP, 'fullfield_placebo.csv')
del Rr, RS, RN

# ======================================================================
# (D) EUROPE Q2: EEA DAMAGE COSTS PER TONNE
# ======================================================================
print('\n' + '=' * 70 + '\n(D) Europe, valuation with EEA damage costs\n'
      + '=' * 70)
# ETC/ATNI Report 2020/4 (Schucht et al. 2021), Table 27: aggregate marginal
# damage costs over EEA38+UK, EUR2019 per tonne: (NOx, PM10, SO2) x
# (VOLY, VSL). PM10 is applied to reported dust.
COST = {
 'Austria': (22542, 75653, 44979, 147529, 34630, 113123),
 'Belgium': (19464, 67390, 103329, 332491, 49322, 159275),
 'Bulgaria': (12010, 39745, 53333, 201070, 15114, 46368),
 'Croatia': (17497, 59756, 35252, 124874, 23315, 79013),
 'Czechia': (15017, 49078, 57202, 183409, 22528, 71277),
 'Denmark': (7595, 25273, 25437, 80593, 17274, 54274),
 'Estonia': (2056, 6947, 5157, 17360, 2242, 6789),
 'Finland': (4101, 14457, 13212, 42445, 5393, 17104),
 'France': (19396, 63549, 42464, 135189, 34580, 111252),
 'Germany': (20447, 74611, 49219, 173147, 34621, 116457),
 'Greece': (7321, 26639, 27156, 94614, 11901, 36550),
 'Hungary': (20348, 68822, 50635, 169836, 24317, 77634),
 'Ireland': (10510, 29780, 12735, 32609, 26844, 77934),
 'Italy': (28857, 108056, 107384, 384838, 26967, 93755),
 'Lithuania': (5158, 18333, 11450, 40353, 7845, 25598),
 'Latvia': (4965, 19089, 17423, 64043, 8276, 28699),
 'Netherlands': (23088, 79355, 61781, 191298, 42521, 135753),
 'Poland': (8457, 26911, 27684, 83938, 14289, 42598),
 'Portugal': (8804, 31399, 43859, 151956, 10555, 35283),
 'Romania': (15800, 52360, 42028, 141119, 20072, 61831),
 'Slovakia': (16217, 50172, 49995, 151580, 19963, 60571),
 'Slovenia': (24889, 84337, 72969, 242259, 28225, 93509),
 'Spain': (12586, 42076, 41425, 130955, 22548, 71915),
 'Sweden': (5453, 18595, 10944, 34765, 6460, 20188),
 'United Kingdom': (16537, 55583, 56373, 174188, 37714, 117415),
}
ALIAS = {'Czech Republic': 'Czechia', 'UK': 'United Kingdom',
         'United Kingdom of Great Britain and Northern Ireland':
         'United Kingdom', 'Slovak Republic': 'Slovakia'}
d = pd.read_csv(f'{DIR}/eea/F5_2_LCP_Energy_Emissions.csv', low_memory=False)
d['featureValue'] = pd.to_numeric(d['featureValue'], errors='coerce')
d['key'] = np.where(d['featureType'] == 'LCPCharacteristics', d['unit'],
                    d['featureType'])
wu = d.pivot_table(index=['LCPInspireId', 'reportingYear'], columns='key',
                   values='featureValue', aggfunc='sum').reset_index().fillna(0)
for c_ in ['SO2', 'NOX', 'DUST']:
    if c_ not in wu:
        wu[c_] = 0.0
ctry = d.groupby('LCPInspireId')['countryName'].first()
EU = pd.read_csv(f'{DIR}/eu_unit_exits.csv')
EU = EU[EU['last_coal_year'].between(2019, 2022)]
first_y = int(d['reportingYear'].min())
rows = []
miss = set()
for x in EU.itertuples():
    g = wu[wu['LCPInspireId'] == x.lcp].set_index('reportingYear').sort_index()
    cn = ALIAS.get(x.country, x.country)
    if cn not in COST:
        miss.add(x.country); continue
    cno, cnv, cpo, cpv, cso, csv_ = COST[cn]
    yrs = [y for y in range(first_y, x.last_coal_year) if y in g.index]
    if x.last_coal_year - 1 not in g.index or not yrs:
        continue
    for basis, (cN, cP, cS) in [('VOLY', (cno, cpo, cso)),
                                ('VSL', (cnv, cpv, csv_))]:
        dm = {y: g.at[y, 'SO2'] * cS + g.at[y, 'NOX'] * cN
              + g.at[y, 'DUST'] * cP for y in yrs}
        rows.append(dict(lcp=x.lcp, name=x.name, country=cn,
                         last_coal_year=x.last_coal_year, basis=basis,
                         mw_th=x.unit_mw_th, dmg_peak=max(dm.values()),
                         dmg_final=dm[x.last_coal_year - 1],
                         so2_peak=g.loc[yrs, 'SO2'].max(),
                         so2_final=g.at[x.last_coal_year - 1, 'SO2']))
EV = pd.DataFrame(rows)
if miss:
    print('No damage cost for:', sorted(miss))
for basis in ['VOLY', 'VSL']:
    e_ = EV[(EV.basis == basis) & (EV.dmg_peak > 0)]
    big = e_[e_.mw_th >= 1000]
    sh = 1 - e_.dmg_final.sum() / e_.dmg_peak.sum()
    print(f'  {basis}: {len(e_)} unit exits 2019-2022; share of the damage '
          f'cut from the 2016-or-later peak before the last full year '
          f'{100 * sh:.0f}%; multiple {e_.dmg_peak.sum() / e_.dmg_final.sum():.2f}'
          f'; damages at peak EUR{e_.dmg_peak.sum() / 1e9:.1f}bn/yr vs '
          f'EUR{e_.dmg_final.sum() / 1e9:.1f}bn in the last full year; units '
          f'>= 1,000 MWth ({len(big)}): {100 * (1 - big.dmg_final.sum() / big.dmg_peak.sum()):.0f}%')
    SUM.append(dict(analysis=f'Europe Q2 damages ({basis})', events=len(e_),
                    mean=sh, multiple=e_.dmg_peak.sum() / e_.dmg_final.sum(),
                    dmg_peak_bn=e_.dmg_peak.sum() / 1e9,
                    dmg_final_bn=e_.dmg_final.sum() / 1e9))
e_ = EV[(EV.basis == 'VOLY') & (EV.so2_peak > 0)]
print(f'  check, SO2 share on this sample: '
      f'{100 * (1 - e_.so2_final.sum() / e_.so2_peak.sum()):.0f}% (58% in '
      'the paper)')
save(EV, 'eu_valuation.csv')
# US over a comparable window: peak within the five years before exit
mvv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
Gg = np.load(f'{DIR}/isrm_grid.npz')
if not (np.allclose(mvv['Location_W'].values, Gg['W'])
        and np.allclose(mvv['Location_S'].values, Gg['S'])):
    key = pd.Series(range(len(mvv)), index=pd.MultiIndex.from_arrays(
        [mvv['Location_W'].round(0), mvv['Location_S'].round(0)]))
    idx = key.reindex(pd.MultiIndex.from_arrays(
        [np.round(Gg['W'], 0), np.round(Gg['S'], 0)])).values
    mvv = mvv.iloc[idx.astype(int)].reset_index(drop=True)
au = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                 usecols=['facilityid', 'unitid', 'year', 'so2mass', 'noxmass'])
for c_ in ['so2mass', 'noxmass']:
    au[c_] = pd.to_numeric(au[c_], errors='coerce').fillna(0)
au['uid'] = au['facilityid'].astype(str) + '_' + au['unitid'].astype(str)
uy = au.groupby(['uid', 'year'])[['so2mass', 'noxmass']].sum()
Ud = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
Ud = Ud[Ud['cell'] >= 0]
pk, fn = 0.0, 0.0
nu_ = 0
for x in Ud.itertuples():
    ms = mvv['MD_SO2_high'].values[int(x.cell)]
    mn = mvv['MD_NOx_high'].values[int(x.cell)]
    vals = [(uy.at[(x.uid, y), 'so2mass'] * ms + uy.at[(x.uid, y), 'noxmass']
             * mn) * ST2T for y in range(x.last_coal_year - 5, x.last_coal_year)
            if (x.uid, y) in uy.index]
    if (x.uid, x.last_coal_year - 1) not in uy.index or not vals:
        continue
    pk += max(vals); fn += vals[-1]          # vals[-1] is year t-1
    nu_ += 1
print(f'  US comparison, InMAP damages, peak within the five years before '
      f'exit ({nu_} units): share before the last full year '
      f'{100 * (1 - fn / pk):.0f}%, multiple {pk / fn:.2f}')
SUM.append(dict(analysis='US damages, five-year peak window', events=nu_,
                mean=1 - fn / pk, multiple=pk / fn))

save(pd.DataFrame(SUM), 'batchD_summary.csv')
print('\nSHA-256:')
for p in saved:
    print(f'  {os.path.basename(p)}  '
          f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
