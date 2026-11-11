# POST-REGISTRATION outcome analysis, part 2 (ground monitors):
#  (A) pre-specified robustness for H3b and H3d(a) (PDF section 6):
#      PM2.5 radius 15 and 25 km; background beyond 75 km; dropping events
#      within 2 years of another exit at the same plant; dropping 2015-2016;
#      weights by number of near monitors; before vs after 2012;
#      secondary samples (SO2 drop >= 1,000 and >= 2,500 t)
#  (B) H3a diagnostics (EXPLORATORY, labelled as such in the paper)
#  (C) H3e without the background requirement (clarification)
#  (D) Holm-Bonferroni with the tests available so far (provisional)
# Uses the same helpers as outcomes1.py (executed from that file).
DIR = '/content/drive/MyDrive/coal_closures'
src = open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0]
exec(src)
B = 1999                      # robustness only; main tests used 9,999

def prep(q, val, min_days, bg_km=50):
    q = q[(q['ndays'] >= min_days) & (q[val] > 0)].copy()
    q['yr'] = q['ym'] // 100
    st = q.groupby('site')[['lat', 'lon', 'state']].first()
    Sx = st[['lat', 'lon']].values
    dc = np.full(len(Sx), np.inf); de = np.full(len(Sx), np.inf)
    for la, lo in fc[['latitude', 'longitude']].values:
        dc = np.minimum(dc, km(Sx[:, 0], Sx[:, 1], la, lo))
    for la, lo in ex[['latitude', 'longitude']].values:
        de = np.minimum(de, km(Sx[:, 0], Sx[:, 1], la, lo))
    st['bg'] = dc > bg_km; st['d_exit'] = de
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
    return dict(st=st, Sx=Sx, yrs=yrs, ann=ann, q=q)

def placebo_full(Pp, radius):
    st, Sx, ann = Pp['st'], Pp['Sx'], Pp['ann']
    allowed = set(st.index[st['d_exit'] > radius])
    D = []
    for c in ctrl.itertuples():
        d = km(Sx[:, 0], Sx[:, 1], c.latitude, c.longitude)
        near = [s for s in st.index[d <= radius] if s in allowed]
        for y in range(2005, 2019):
            v = [(ann.at[(s, y - 1), 'rel'] - ann.at[(s, y + 1), 'rel'],
                  ann.at[(s, y - 1), 'lv']) for s in near
                 if (s, y - 1) in ann.index and (s, y + 1) in ann.index]
            if v:
                D.append((c.facilityid, y, np.mean([x[0] for x in v]),
                          np.mean([x[1] for x in v])))
    return pd.DataFrame(D, columns=['fid', 'year', 'd', 'lv'])

def build(Pp, radius, ev, ug):
    """Observed control-adjusted falls for events ev at this design."""
    D = placebo_full(Pp, radius)
    ym, pool = D.groupby('year')['d'].mean().to_dict(), D['d'].mean()
    sd = (D['d'] * D['lv']).std() if ug else D['d'].std()
    o = [change(Pp, r.lat, r.lon, r.event_year, radius)
         for r in ev.itertuples()]
    ev = ev.copy()
    ev['dlog'], ev['pre_level'], ev['n_obs'] = zip(*o)
    ev = ev[ev['n_obs'] > 0].copy()
    adj = ev['event_year'].map(ym).fillna(pool)
    ev['fall'] = (ev['dlog'] - adj) * (ev['pre_level'] if ug else 1)
    return ev, sd, D['fid'].nunique()

def near_other_exit(r):
    return any(0 < abs(e - r.event_year) <= 2
               for e in exit_years.get(int(r.fid), set()))

def weighted(label, ev, sd):
    w = ev['n_obs'].values.astype(float); y = ev['fall'].values
    m = (w * y).sum() / w.sum()
    gs = pd.Series(w).groupby(ev['fid'].values).sum().values
    se = sd * np.sqrt((gs ** 2).sum()) / w.sum()
    df = pd.DataFrame({'fid': ev['fid'].values, 'y': y, 'w': w})
    bm = cluster_boot(df, lambda d: (d['w'] * d['y']).sum() / d['w'].sum())
    return dict(test=label, events=len(ev), plants=ev['fid'].nunique(),
                mean=m, se_placebo=se,
                ci_cluster_boot=(np.percentile(bm, 2.5),
                                 np.percentile(bm, 97.5)),
                ci_placebo=(m - 1.96 * se, m + 1.96 * se),
                p0_primary=2 * (1 - Phi(abs(m / se))),
                p_alt_primary=Phi((m - 1.1) / se))

rows = []
def run(label, ev, sd, alt):
    if len(ev) < 3:
        print(f'{label}: {len(ev)} events, skipped'); return
    r = summarise(label, ev['fall'].values, ev['fid'].values, sd, alt)
    rows.append(r); show(r)

E2 = E.copy()
base = E2[E2['so2_drop'] >= 5000]

# ================= (A) PM2.5 robustness =================
print('\n=== (A) PRE-SPECIFIED ROBUSTNESS: total PM2.5 (H3b; µg/m³) ===')
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
PM = prep(q, 'pm', 8)
ev35, sd35, _ = build(PM, 35, base, True)
for rad in [15, 25]:
    ev_r, sd_r, npl = build(PM, rad, base, True)
    run(f'PM2.5 radius {rad} km (placebo from {npl} plants)', ev_r, sd_r,
        1.1)
PM75 = prep(q, 'pm', 8, bg_km=75); del q
ev_b, sd_b, _ = build(PM75, 35, base, True)
run('PM2.5 background beyond 75 km', ev_b, sd_b, 1.1)
run('PM2.5 dropping events within 2 years of another exit at the plant',
    ev35[[not near_other_exit(r) for r in ev35.itertuples()]], sd35, 1.1)
run('PM2.5 dropping 2015-2016 (MATS)',
    ev35[~ev35['event_year'].isin([2015, 2016])], sd35, 1.1)
r = weighted('PM2.5 weighted by number of near monitors', ev35, sd35)
rows.append(r); show(r)
run('PM2.5 events before 2012', ev35[ev35['event_year'] < 2012], sd35, 1.1)
run('PM2.5 events 2012 and later', ev35[ev35['event_year'] >= 2012], sd35,
    1.1)
for t in [1000, 2500]:
    ev_t, _, _ = build(PM, 35, E2[E2['so2_drop'] >= t], True)
    run(f'PM2.5 secondary sample, SO2 drop >= {t:,} t', ev_t, sd35, 1.1)

# ================= (B) H3a diagnostics (exploratory) =================
print('\n=== (B) H3a DIAGNOSTICS (EXPLORATORY) ===')
F = pd.read_csv(f'{OUT}/events_pm25_35km.csv')
F['contrib'] = F['pred_main'] * F['fall']
print('Events with the largest predictions:')
print(F.sort_values('pred_main', ascending=False)
      [['name', 'event_year', 'so2_drop', 'layer', 'pred_main', 'fall',
        'fall_raw', 'n_obs']].head(12).round(3).to_string(index=False))
lo = {f: slope0(F.loc[F.fid != f, 'pred_main'], F.loc[F.fid != f, 'fall'])
      for f in F['fid'].unique()}
print(f"Leave-one-plant-out slopes: min {min(lo.values()):.2f}, max "
      f"{max(lo.values()):.2f}")
big = max(lo, key=lambda f: lo[f])
print(f"  dropping {F.loc[F.fid == big, 'name'].iloc[0]} gives the largest "
      f"slope ({lo[big]:.2f})")
print(f"Slope on unadjusted falls: {slope0(F['pred_main'], F['fall_raw']):.2f}")
from scipy.stats import spearmanr
rho, p = spearmanr(F['pred_main'], F['fall'])
print(f'Spearman rank correlation, prediction vs fall: {rho:.2f} (p {p:.3f})')
print(f"SD of predictions {F['pred_main'].std():.3f} µg/m³ vs SD of observed "
      f"falls {F['fall'].std():.3f}: the prediction range is "
      f"{F['pred_main'].std() / F['fall'].std():.0%} of the noise")

# ================= (A) sulfate robustness =================
print('\n=== (A) PRE-SPECIFIED ROBUSTNESS: sulfate, 100 km (H3d(a); log) ===')
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
SO = prep(q, 'so4', 3)
SO75 = prep(q, 'so4', 3, bg_km=75); del q
evs, sds, _ = build(SO, 100, base, False)
ev_b, sd_b, _ = build(SO75, 100, base, False)
run('Sulfate background beyond 75 km', ev_b, sd_b, None)
run('Sulfate dropping events within 2 years of another exit',
    evs[[not near_other_exit(r) for r in evs.itertuples()]], sds, None)
run('Sulfate dropping 2015-2016 (MATS)',
    evs[~evs['event_year'].isin([2015, 2016])], sds, None)
r = weighted('Sulfate weighted by number of near monitors', evs, sds)
r.pop('p_alt_primary'); rows.append(r); show(r)
run('Sulfate events before 2012', evs[evs['event_year'] < 2012], sds, None)
run('Sulfate events 2012 and later', evs[evs['event_year'] >= 2012], sds,
    None)
for t in [1000, 2500]:
    ev_t, _, _ = build(SO, 100, E2[E2['so2_drop'] >= t], False)
    run(f'Sulfate secondary sample, SO2 drop >= {t:,} t', ev_t, sds, None)

# ================= (C) H3e without background =================
print('\n=== (C) H3e WITHOUT THE BACKGROUND REQUIREMENT (clarification) ===')
q = pd.concat([pd.read_csv(f'{DIR}/so2/so2_monthly_{y}.csv')
               for y in range(2004, 2025)])
q = q[(q['ndays'] >= 8) & (q['so2'] > 0)].copy(); q['yr'] = q['ym'] // 100
st2 = q.groupby('site')[['lat', 'lon']].first()
an2 = q.groupby(['site', 'yr']).agg(v=('so2', 'mean'), n=('so2', 'size'))
an2 = an2[an2['n'] >= 8]
out = []
for r_ in F.itertuples():
    d = km(st2['lat'].values, st2['lon'].values, r_.lat, r_.lon)
    v = [np.log(an2.at[(s, r_.event_year - 1), 'v'])
         - np.log(an2.at[(s, r_.event_year + 1), 'v'])
         for s in st2.index[d <= 25]
         if (s, r_.event_year - 1) in an2.index
         and (s, r_.event_year + 1) in an2.index]
    if v:
        out.append((r_.name, r_.event_year, len(v), np.mean(v)))
C2 = pd.DataFrame(out, columns=['name', 'event_year', 'monitors', 'so2_dlog'])
print(f"{len(C2)} events with an SO2 monitor within 25 km; "
      f"{(C2.so2_dlog > 0).sum()} show a fall; median fall "
      f"{C2.so2_dlog.median():.2f} log points "
      f"({100 * (1 - np.exp(-C2.so2_dlog.median())):.0f}%)")
print('  no fall:', ', '.join(f'{a} {b}' for a, b in
                              C2.loc[C2.so2_dlog <= 0,
                                     ['name', 'event_year']].values) or 'none')
C2.to_csv(f'{OUT}/events_so2_nobackground.csv', index=False)

# ================= (D) Holm (provisional) =================
print('\n=== (D) HOLM-BONFERRONI, PROVISIONAL (H3f(a) and H4 pending) ===')
R1 = pd.read_csv(f'{OUT}/results_part1.csv')
def g(test, col):
    s = R1.loc[R1['test'].str.startswith(test), col]
    return float(s.iloc[0])
fam = {'H3b': max(g('H3b/H3c  total', 'p_alt_primary'), 1 / 9999),
       'H3c': g('H3b/H3c  total', 'p0_primary'),
       'H3d(a)': g('H3d(a)  sulfate, 100', 'p0_primary'),
       'H3d(b)': g('H3d(b)  sulfate slope, 100', 'p_slope_eq_1')}
print('Primary (most conservative) p-values:',
      {k: round(v, 4) for k, v in fam.items()})
print('Holm is applied once H3f(a) and H4 are in (six tests in all).')
pd.DataFrame(rows).to_csv(f'{OUT}/results_part2_robustness.csv', index=False)
print('\nSaved results_part2_robustness.csv and '
      'events_so2_nobackground.csv to', OUT)
