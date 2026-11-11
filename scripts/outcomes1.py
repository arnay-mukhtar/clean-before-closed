# POST-REGISTRATION outcome analysis, part 1 (ground monitors).
# Registered on OSF 9 Nov 2026 before this script was run.
# Runs, exactly as pre-registered (PDF sections 5-6):
#   H3b, H3c, H3a (+ pre-specified heterogeneity)  total PM2.5, 35 km
#   H3d(a), H3d(b)                                  sulfate, 100 km (+50 km)
#   H3e                                             SO2 check, 25 km, and
#                                                   H3a-H3d without failures
# Inference: placebo events at never-exit plants (clustered by plant),
# cluster bootstrap by plant, wild cluster bootstrap (Rademacher, 9,999),
# placebo dates at the event plants. Most conservative p is primary.
# Step 0 checks every frozen input against fingerprints.csv and stops on
# any mismatch. Results go to MyDrive/coal_closures/outcomes/.
DIR = '/content/drive/MyDrive/coal_closures'
import os, hashlib
import numpy as np, pandas as pd
from math import erf
from pyproj import Transformer

OUT = f'{DIR}/outcomes'
os.makedirs(OUT, exist_ok=True)
Phi = lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))
TPY = 907184.74e6 / (365.25 * 86400)
B = 9999
SEED = 20261109

# ---------- 0. integrity check ----------
fpr = pd.read_csv(f'{DIR}/fingerprints.csv').set_index('file')['sha256']
need = ['final_pm25_35km_predictions.csv', 'final_pm25_35km_placebo.csv',
        'final_so4_100km_predictions.csv', 'final_so4_100km_placebo.csv',
        'final_so4_50km_predictions.csv', 'final_so4_50km_placebo.csv',
        'so2_check_final.csv', 'model_predictions_us_events_v2.csv',
        'us_plant_exit_events.csv', 'us_unit_exits.csv',
        'us_epa_annual_all.csv', 'isrm_coal_facility_cells.csv']
bad = [f for f in need if hashlib.sha256(open(f'{DIR}/{f}', 'rb').read())
       .hexdigest() != fpr.get(f)]
if bad:
    raise SystemExit(f'STOP: these files differ from the registered '
                     f'fingerprints: {bad}')
print('Integrity check passed: all', len(need), 'frozen inputs match the '
      'registered fingerprints.\n')

# ---------- shared setup (identical to finalize.py) ----------
def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))

fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv')
G = np.load(f'{DIR}/isrm_grid.npz')
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
W, Ee, S, N = G['W'], G['E'], G['S'], G['N']
def cell_of(lat, lon):
    x, y = tr.transform(lon, lat)
    h = np.where((W <= x) & (x < Ee) & (S <= y) & (y < N))[0]
    return int(h[0]) if len(h) else -1

E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ex = fc[fc['facilityid'].isin(exit_fids)]
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
cy = a[a['primaryfuelinfo'].fillna('').str.contains('Coal')
       & (a['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
ctrl = fc[fc['facilityid'].isin([f for f, ys in cy.items()
                                 if set(range(2003, 2020)) <= ys
                                 and f not in exit_fids])]
exit_years = {}                            # every exit year at each plant
for f, y in zip(E['fid'], E['event_year']):
    exit_years.setdefault(int(f), set()).add(int(y))
for uid, y in zip(U['uid'], U['last_coal_year']):
    exit_years.setdefault(int(str(uid).split('_')[0]), set()).add(int(y))

def prep(q, val, min_days):
    q = q[(q['ndays'] >= min_days) & (q[val] > 0)].copy()
    q['yr'] = q['ym'] // 100
    st = q.groupby('site')[['lat', 'lon', 'state']].first()
    Sx = st[['lat', 'lon']].values
    dc = np.full(len(Sx), np.inf); de = np.full(len(Sx), np.inf)
    for la, lo in fc[['latitude', 'longitude']].values:
        dc = np.minimum(dc, km(Sx[:, 0], Sx[:, 1], la, lo))
    for la, lo in ex[['latitude', 'longitude']].values:
        de = np.minimum(de, km(Sx[:, 0], Sx[:, 1], la, lo))
    st['bg'] = dc > 50; st['d_exit'] = de
    st['cell'] = [cell_of(x, y) for x, y in Sx]
    ny = q.groupby(['site', 'yr']).size()
    yrs = ny[ny >= 8].reset_index().groupby('site')['yr'].apply(set)
    q['l'] = np.log(q[val])
    bgm = (q[q['site'].isin(set(st.index[st['bg']]))]
           .groupby(['state', 'ym'])['l'].mean().rename('bgl'))
    qq = q.join(bgm, on=['state', 'ym']).dropna(subset=['bgl'])
    qq['rel'] = qq['l'] - qq['bgl']
    ann = qq.groupby(['site', 'yr']).agg(rel=('rel', 'mean'),
                                         n=('rel', 'size'),
                                         lv=(val, 'mean'))
    ann = ann[ann['n'] >= 8]
    return dict(st=st, Sx=Sx, yrs=yrs, ann=ann)

def change(P, lat, lon, y, radius, need_cell=True):
    """Pre-minus-post change in relative log concentration at year y,
    near monitors chosen exactly as for the frozen predictions."""
    st, Sx, yrs, ann = P['st'], P['Sx'], P['yrs'], P['ann']
    b, a_ = y - 1, y + 1
    d = km(Sx[:, 0], Sx[:, 1], lat, lon)
    v, l = [], []
    for s in st.index[d <= radius]:
        if not {b, a_} <= yrs.get(s, set()):
            continue
        if need_cell and st.at[s, 'cell'] < 0:
            continue
        bg = [x for x in st.index[(st['state'] == st.at[s, 'state'])
                                  & st['bg']
                                  & ((st['cell'] >= 0) if need_cell else True)]
              if {b, a_} <= yrs.get(x, set())]
        if not bg or (s, b) not in ann.index or (s, a_) not in ann.index:
            continue
        v.append(ann.at[(s, b), 'rel'] - ann.at[(s, a_), 'rel'])
        l.append(ann.at[(s, b), 'lv'])
    if not v:
        return np.nan, np.nan, 0
    return float(np.mean(v)), float(np.mean(l)), len(v)

def placebo_by_year(P, radius):
    """Placebo changes at never-exit plants (as in finalize.py), by year."""
    st, Sx, ann = P['st'], P['Sx'], P['ann']
    allowed = set(st.index[st['d_exit'] > radius])
    D = []
    for c in ctrl.itertuples():
        d = km(Sx[:, 0], Sx[:, 1], c.latitude, c.longitude)
        near = [s for s in st.index[d <= radius] if s in allowed]
        for y in range(2005, 2019):
            v = [ann.at[(s, y - 1), 'rel'] - ann.at[(s, y + 1), 'rel']
                 for s in near
                 if (s, y - 1) in ann.index and (s, y + 1) in ann.index]
            if v:
                D.append((c.facilityid, y, np.mean(v)))
    D = pd.DataFrame(D, columns=['fid', 'year', 'd'])
    return D.groupby('year')['d'].mean().to_dict(), D['d'].mean()

# ---------- inference helpers ----------
rng = np.random.default_rng(SEED)

def clustered_se(sd, groups):
    n = np.bincount(pd.factorize(groups)[0])
    return sd * np.sqrt((n ** 2).sum()) / n.sum()

def cluster_boot(df, stat):
    g = df.groupby('fid').indices
    keys = list(g)
    out = np.empty(B)
    for i in range(B):
        pick = rng.choice(len(keys), len(keys))
        idx = np.concatenate([g[keys[k]] for k in pick])
        out[i] = stat(df.iloc[idx])
    return out

def wild_p(y, groups, mu0, one_sided_less=False):
    """Wild cluster bootstrap (Rademacher, null imposed) for a mean."""
    y = np.asarray(y, float)
    gi = pd.factorize(groups)[0]; Gn = gi.max() + 1
    def tstat(v):
        m = v.mean(); u = np.bincount(gi, v - m)
        se = np.sqrt((u ** 2).sum()) / len(v) * np.sqrt(Gn / (Gn - 1))
        return (m - mu0) / se
    t0 = tstat(y); u = y - mu0
    ts = np.empty(B)
    for i in range(B):
        w = rng.choice([-1.0, 1.0], Gn)[gi]
        ts[i] = tstat(mu0 + w * u)
    if one_sided_less:
        return t0, (ts <= t0).mean()
    return t0, (np.abs(ts) >= abs(t0)).mean()

def slope0(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x * y).sum() / (x * x).sum()

def boot_slope_p(bs, value=1.0):
    return min(1.0, 2 * min((bs <= value).mean(), (bs >= value).mean()))

def placebo_dates(P, ev, radius, to_ug, need_cell=True):
    """Same statistic at the event plants in years >= 3 from any exit there."""
    cand = {}
    for r in ev.drop_duplicates('fid').itertuples():
        ys = exit_years.get(r.fid, set())
        c = []
        for y in range(2001, 2025):
            if all(abs(y - e) >= 3 for e in ys):
                dl, lv, n = change(P, r.lat, r.lon, y, radius, need_cell)
                if n:
                    c.append(dl * lv if to_ug else dl)
        cand[r.fid] = np.array(c)
    use = ev[ev['fid'].map(lambda f: len(cand[f]) > 0)]
    draws = np.empty(B)
    for i in range(B):
        draws[i] = np.mean([rng.choice(cand[f]) for f in use['fid']])
    return use, draws

def summarise(label, y, groups, sd_placebo, mu_alt=None, ev_pd=None,
              obs_pd=None, draws=None):
    """y: control-adjusted event falls. Returns a results dict."""
    m = float(np.mean(y)) if len(y) else np.nan; n = len(y)
    if n < 3 or len(set(groups)) < 2:
        print(f'\n{label}: only {n} events, not estimated')
        return dict(test=label, events=n)
    se = clustered_se(sd_placebo, groups)
    res = dict(test=label, events=n, plants=len(set(groups)), mean=m,
               se_placebo=se, ci_placebo=(m - 1.96 * se, m + 1.96 * se))
    df = pd.DataFrame({'fid': groups, 'y': y})
    bm = cluster_boot(df, lambda d: d['y'].mean())
    res['ci_cluster_boot'] = (np.percentile(bm, 2.5), np.percentile(bm, 97.5))
    res['p0_placebo'] = 2 * (1 - Phi(abs(m / se)))
    res['p0_wild'] = wild_p(y, groups, 0.0)[1]
    if mu_alt is not None:
        res['p_alt_placebo'] = Phi((m - mu_alt) / se)
        res['p_alt_wild'] = wild_p(y, groups, mu_alt, True)[1]
    if draws is not None:
        est = obs_pd - draws.mean()
        res['pd_events'] = len(ev_pd); res['pd_estimate'] = est
        res['ci_placebo_dates'] = (obs_pd - np.percentile(draws, 97.5),
                                   obs_pd - np.percentile(draws, 2.5))
        c = draws - draws.mean()
        res['p0_placebo_dates'] = (np.abs(c) >= abs(est)).mean()
        if mu_alt is not None:
            res['p_alt_placebo_dates'] = (c <= est - mu_alt).mean()
    res['p0_primary'] = max(v for k, v in res.items()
                            if k.startswith('p0_'))
    if mu_alt is not None:
        res['p_alt_primary'] = max(v for k, v in res.items()
                                   if k.startswith('p_alt_'))
    return res

def show(r):
    f = lambda t: f'[{t[0]:.3f}, {t[1]:.3f}]'
    print(f"\n{r['test']}: {r['events']} events at {r['plants']} plants")
    print(f"  mean (control-adjusted) {r['mean']:.3f}; placebo SE "
          f"{r['se_placebo']:.3f}; 95% CI placebo {f(r['ci_placebo'])}, "
          f"cluster bootstrap {f(r['ci_cluster_boot'])}"
          + (f", placebo dates {f(r['ci_placebo_dates'])} "
             f"({r['pd_events']} events, estimate {r['pd_estimate']:.3f})"
             if 'ci_placebo_dates' in r else ''))
    print('  p vs 0 (two-sided): ' + ', '.join(
        f"{k[3:]} {v:.4f}" for k, v in r.items() if k.startswith('p0_')))
    if 'p_alt_primary' in r:
        print('  p for mean < 1.1 (one-sided): ' + ', '.join(
            f"{k[6:]} {v:.4f}" for k, v in r.items()
            if k.startswith('p_alt_')))

results, events_out = [], {}

# ================= TOTAL PM2.5, 35 km (H3a, H3b, H3c) =================
print('Reading AQS PM2.5 ...', flush=True)
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
PM = prep(q, 'pm', 8); del q
F = pd.read_csv(f'{DIR}/final_pm25_35km_predictions.csv')
F = F[(F['n_near_bg'] > 0) & (F['so2_drop'] >= 5000)]
F = F.merge(E[['fid', 'event_year', 'lat', 'lon']], on=['fid', 'event_year'])
ym_pm, pool_pm = placebo_by_year(PM, 35)
obs = [change(PM, r.lat, r.lon, r.event_year, 35) for r in F.itertuples()]
F['dlog'], F['pre_level'], F['n_obs'] = zip(*obs)
F = F[F['n_obs'] > 0].copy()
F['adj'] = F['event_year'].map(ym_pm).fillna(pool_pm)
F['adj_from_pool'] = ~F['event_year'].isin(ym_pm)
F['fall_raw'] = F['dlog'] * F['pre_level']
F['fall'] = (F['dlog'] - F['adj']) * F['pre_level']
pl = pd.read_csv(f'{DIR}/final_pm25_35km_placebo.csv')
sd_pm = (pl['d'] * pl['pre_level']).std()
print(f'Primary PM2.5 sample: {len(F)} events '
      f'({int(F.adj_from_pool.sum())} after 2018 use the pooled placebo mean '
      f'{pool_pm:.4f}, as no year-specific placebo exists)')
evpd, dr = placebo_dates(PM, F, 35, True)
obs_pd = evpd['fall_raw'].mean()
r = summarise('H3b/H3c  total PM2.5, 35 km', F['fall'].values,
              F['fid'].values, sd_pm, 1.1, evpd, obs_pd, dr)
r['mean_prediction'] = F['pred_main'].mean()
results.append(r); show(r)
print(f"  mean frozen prediction {r['mean_prediction']:.3f} µg/m³; "
      f"unadjusted mean {F['fall_raw'].mean():.3f}")

# H3a slope and pre-specified heterogeneity
b = slope0(F['pred_main'], F['fall'])
bs = cluster_boot(F, lambda d: slope0(d['pred_main'], d['fall']))
r3a = dict(test='H3a  slope of observed on predicted PM2.5', events=len(F),
           slope=b, ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
           p_slope_eq_1=boot_slope_p(bs))
results.append(r3a)
print(f"\nH3a: slope {b:.2f}, cluster-bootstrap 95% CI "
      f"[{r3a['ci'][0]:.2f}, {r3a['ci'][1]:.2f}], p(slope = 1) "
      f"{r3a['p_slope_eq_1']:.4f}")
for tag in ['high', 'low']:
    print(f"  sensitivity, {tag}-stack prediction: slope "
          f"{slope0(F['pred_' + tag], F['fall']):.2f}")
F['pred_noppm'] = F['so4_main'] + F['no3_main']
print(f"  sensitivity, no primary PM2.5: slope "
      f"{slope0(F['pred_noppm'], F['fall']):.2f}")
F['low'] = (F['layer'] == 1).astype(float)
F['pshare'] = (F['ppm_main'] / F['pred_main']).replace([np.inf, -np.inf],
                                                        np.nan).fillna(0)
def het(d):
    x = d['pred_main'].values
    X = np.column_stack([x, x * d['low'].values,
                         x * (d['pshare'].values - F['pshare'].mean())])
    return np.linalg.lstsq(X, d['fall'].values, rcond=None)[0]
if F['low'].sum() >= 3 and F['low'].sum() <= len(F) - 3:
    h0 = het(F)
    hb = np.array([het(F.iloc[np.concatenate(
        [np.where(F['fid'].values == f)[0]
         for f in rng.choice(F['fid'].unique(), F['fid'].nunique())])])
        for _ in range(2000)])
    print(f"  heterogeneity (predicted signs +, +): low-layer interaction "
          f"{h0[1]:.2f} [{np.percentile(hb[:, 1], 2.5):.2f}, "
          f"{np.percentile(hb[:, 1], 97.5):.2f}]; primary-share interaction "
          f"{h0[2]:.2f} [{np.percentile(hb[:, 2], 2.5):.2f}, "
          f"{np.percentile(hb[:, 2], 97.5):.2f}]")
else:
    print(f"  heterogeneity: only {int(F['low'].sum())} low-layer events; "
          'layer interaction not estimable, reported as such')
events_out['pm25_35km'] = F

# Decision rule (pre-registered table)
lo, hi = r['ci_placebo']; pr = r['mean_prediction']
for name, (lo, hi) in [('placebo', r['ci_placebo']),
                       ('cluster bootstrap', r['ci_cluster_boot']),
                       ('placebo dates', r['ci_placebo_dates'])]:
    if lo <= pr <= hi and hi < 1.1:
        v = 'model consistent; published ex-post magnitudes rejected'
    elif lo > pr and hi >= 1.1:
        v = 'model underpredicts; published magnitudes supported'
    elif lo > pr and hi < 1.1:
        v = 'model underpredicts; published magnitudes too large'
    else:
        v = 'inconclusive'
    print(f'  decision rule ({name} interval): {v}')

# ================= SULFATE (H3d) =================
print('\nReading AQS sulfate ...', flush=True)
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
SO = prep(q, 'so4', 3); del q
for rad in [100, 50]:
    Fs = pd.read_csv(f'{DIR}/final_so4_{rad}km_predictions.csv')
    Fs = Fs[(Fs['n_near_bg'] > 0) & (Fs['so2_drop'] >= 5000)
            & Fs['pred_rel_main'].notna()]
    Fs = Fs.merge(E[['fid', 'event_year', 'lat', 'lon']],
                  on=['fid', 'event_year'])
    ym_s, pool_s = placebo_by_year(SO, rad)
    obs = [change(SO, r_.lat, r_.lon, r_.event_year, rad)
           for r_ in Fs.itertuples()]
    Fs['dlog'], Fs['pre_level'], Fs['n_obs'] = zip(*obs)
    Fs = Fs[Fs['n_obs'] > 0].copy()
    Fs['adj'] = Fs['event_year'].map(ym_s).fillna(pool_s)
    Fs['fall_raw'] = Fs['dlog']
    Fs['fall'] = Fs['dlog'] - Fs['adj']
    sd_s = pd.read_csv(f'{DIR}/final_so4_{rad}km_placebo.csv')['d'].std()
    evs, drs = placebo_dates(SO, Fs, rad, False)
    tag = 'H3d(a)  sulfate, 100 km (log points)' if rad == 100 else \
        'H3d(a) robustness  sulfate, 50 km (log points)'
    rs = summarise(tag, Fs['fall'].values, Fs['fid'].values, sd_s, None,
                   evs, evs['fall_raw'].mean(), drs)
    rs['mean_prediction'] = Fs['pred_rel_main'].mean()
    results.append(rs); show(rs)
    print(f"  mean predicted relative fall {rs['mean_prediction']:.4f}")
    b = slope0(Fs['pred_rel_main'], Fs['fall'])
    bs = cluster_boot(Fs, lambda d: slope0(d['pred_rel_main'], d['fall']))
    rd = dict(test=f'H3d(b)  sulfate slope, {rad} km', events=len(Fs),
              slope=b, ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
              p_slope_eq_1=boot_slope_p(bs))
    results.append(rd)
    print(f"  H3d(b): slope {b:.2f}, 95% CI [{rd['ci'][0]:.2f}, "
          f"{rd['ci'][1]:.2f}], p(slope = 1) {rd['p_slope_eq_1']:.4f}")
    events_out[f'so4_{rad}km'] = Fs

# ================= SO2 CHECK (H3e) =================
print('\nReading AQS SO2 ...', flush=True)
q = pd.concat([pd.read_csv(f'{DIR}/so2/so2_monthly_{y}.csv')
               for y in range(2004, 2025)])
S2 = prep(q, 'so2', 8); del q
C = F[['fid', 'name', 'event_year', 'lat', 'lon']].copy()
obs = [change(S2, r_.lat, r_.lon, r_.event_year, 25, need_cell=False)
       for r_ in C.itertuples()]
C['so2_dlog'], _, C['so2_monitors'] = zip(*obs)
C = C[C['so2_monitors'] > 0]
C['passes'] = C['so2_dlog'] > 0
fails = C[~C['passes']]
print(f"H3e: {len(C)} events with an SO2 monitor; {int(C.passes.sum())} "
      f"show an SO2 fall; mean fall {C.so2_dlog.mean():.3f} log points")
if len(fails):
    print('  events without an SO2 fall:')
    print(fails[['name', 'event_year', 'so2_dlog']].to_string(index=False))
    drop = set(zip(fails['fid'], fails['event_year']))
    Fx = F[[k not in drop for k in zip(F['fid'], F['event_year'])]]
    rx = summarise('H3b/H3c excluding H3e failures', Fx['fall'].values,
                   Fx['fid'].values, sd_pm, 1.1)
    results.append(rx)
    if 'mean' in rx: show(rx)
    print(f"  H3a slope excluding failures: "
          f"{slope0(Fx['pred_main'], Fx['fall']):.2f}")
    Fs = events_out['so4_100km']
    Fsx = Fs[[k not in drop for k in zip(Fs['fid'], Fs['event_year'])]]
    sd_s = pd.read_csv(f'{DIR}/final_so4_100km_placebo.csv')['d'].std()
    rsx = summarise('H3d(a) excluding H3e failures', Fsx['fall'].values,
                    Fsx['fid'].values, sd_s)
    results.append(rsx)
    if 'mean' in rsx: show(rsx)
events_out['so2_check'] = C

# ---------- save ----------
for k, d in events_out.items():
    d.to_csv(f'{OUT}/events_{k}.csv', index=False)
pd.DataFrame(results).to_csv(f'{OUT}/results_part1.csv', index=False)
print('\nSaved event-level results and results_part1.csv to', OUT)
