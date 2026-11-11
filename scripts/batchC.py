# POST-REGISTRATION robustness for the registered tests and Q2 (batch C,
# 10 Nov 2026), answering the independent referee report. No new outcome
# definitions; every statistic is the registered one.
#  (1) Inference that carries the control-adjustment error: joint cluster
#      bootstrap over event plants AND placebo plants (year-specific placebo
#      means recomputed in every draw).
#  (2) Other dependence structures: cluster-robust SEs and wild bootstrap p
#      with clusters = plant, state-by-event-year, state, and groups of
#      events that share a near monitor.
#  (3) Leave-one-state-out.
#  (4) Unadjusted sulfate estimate.
#  (5) What drives the control adjustment: placebo plant-year changes
#      regressed on the placebo plant's own SO2 change.
#  (6) H3a and H3d(b) slopes without the two H3e failures (registered).
#  (7) Q2 damages with the low release layer (registered sensitivity; the
#      high layer is the main specification).
DIR = '/content/drive/MyDrive/coal_closures'
exec(open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0])
# names W, Ee, S, N, G, E, U, a, ex, fc are used by the helpers: do not reuse
import hashlib
NBC = 4999
rngc = np.random.default_rng(20261111)
saved = []
def save(df, name):
    p = f'{OUT}/{name}'; df.to_csv(p, index=False); saved.append(p)

# ---------- helpers ----------
def placebo_changes(P, radius):
    """Placebo plant-year changes as in placebo_by_year, kept by plant."""
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
            lv = [ann.at[(s, y - 1), 'lv'] for s in near
                  if (s, y - 1) in ann.index and (s, y + 1) in ann.index]
            if v:
                D.append((c.facilityid, y, np.mean(v), np.mean(lv)))
    return pd.DataFrame(D, columns=['fid', 'year', 'd', 'lv'])

def near_sites(P, lat, lon, y, radius, need_cell=True):
    """The near monitors that change() uses for this event (same rules)."""
    st, Sx, yrs, ann = P['st'], P['Sx'], P['yrs'], P['ann']
    b, a_ = y - 1, y + 1
    d = km(Sx[:, 0], Sx[:, 1], lat, lon)
    out = []
    for s in st.index[d <= radius]:
        if not {b, a_} <= yrs.get(s, set()):
            continue
        if need_cell and st.at[s, 'cell'] < 0:
            continue
        bg = [x for x in st.index[(st['state'] == st.at[s, 'state'])
                                  & st['bg']
                                  & ((st['cell'] >= 0) if need_cell else True)]
              if {b, a_} <= yrs.get(x, set())]
        if bg and (s, b) in ann.index and (s, a_) in ann.index:
            out.append(s)
    return out

def components(ev, sites):
    """Union-find: events joined if they share a plant or a near monitor."""
    par = list(range(len(ev)))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    owner = {}
    for i, (fid, ss) in enumerate(zip(ev['fid'], sites)):
        for key in [('p', fid)] + [('s', s) for s in ss]:
            if key in owner:
                par[f(i)] = f(owner[key])
            else:
                owner[key] = i
    return np.array([f(i) for i in range(len(ev))])

def cr_se(y, groups):
    y = np.asarray(y, float); gi = pd.factorize(groups)[0]
    G = gi.max() + 1; u = np.bincount(gi, y - y.mean())
    return np.sqrt((u ** 2).sum()) / len(y) * np.sqrt(G / (G - 1)), G

def joint_boot(ev, D, to_ug):
    """Resample event plants and placebo plants together; the year-specific
    placebo means (and the pooled fallback) are recomputed in every draw."""
    ei_, eu = pd.factorize(ev['fid']); di_, du = pd.factorize(D['fid'])
    dl, lv = ev['dlog'].values, ev['pre_level'].values
    eyr = ev['event_year'].values.astype(int)
    dyr = D['year'].values.astype(int); dd = D['d'].values
    ny = max(eyr.max(), dyr.max()) + 1
    out = np.empty(NBC)
    for b in range(NBC):
        we = np.bincount(rngc.integers(0, len(eu), len(eu)),
                         minlength=len(eu))[ei_]
        wd = np.bincount(rngc.integers(0, len(du), len(du)),
                         minlength=len(du))[di_]
        num = np.bincount(dyr, wd * dd, minlength=ny)
        den = np.bincount(dyr, wd, minlength=ny)
        pool = (wd * dd).sum() / wd.sum()
        ym = np.where(den > 0, num / np.maximum(den, 1e-12), pool)
        f = dl - ym[eyr]
        if to_ug:
            f = f * lv
        out[b] = (we * f).sum() / we.sum()
    return out

def ptwo(bs, v=0.0):
    return min(1.0, 2 * min((bs <= v).mean(), (bs >= v).mean()))

# ---------- data ----------
Ex = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')[['fid', 'event_year',
                                                      'state']]
def with_state(d):
    return d if 'state' in d else d.merge(Ex, on=['fid', 'event_year'],
                                          how='left')
F = with_state(pd.read_csv(f'{OUT}/events_pm25_35km.csv'))
SV = with_state(pd.read_csv(f'{OUT}/events_so4_100km.csv'))
assert F['state'].notna().all() and SV['state'].notna().all()
print(f'{len(F)} PM2.5 events, {len(SV)} sulfate events (expected 51 and 44)')
print('Reading AQS PM2.5 and sulfate ...', flush=True)
q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
PM = prep(q, 'pm', 8); del q
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
SO = prep(q, 'so4', 3); del q
Dpm, Dso = placebo_changes(PM, 35), placebo_changes(SO, 100)
print(f'placebo plant-years: PM2.5 {len(Dpm)} at {Dpm.fid.nunique()} plants '
      f'(mean {Dpm.d.mean():.4f}); sulfate {len(Dso)} at '
      f'{Dso.fid.nunique()} plants (mean {Dso.d.mean():.4f})')

rows = []
for lab, ev, P, D, rad, ug in [('PM2.5 35 km (ug/m3)', F, PM, Dpm, 35, True),
                               ('sulfate 100 km (log points)', SV, SO, Dso,
                                100, False)]:
    print('\n' + '=' * 70 + f'\n{lab}\n' + '=' * 70, flush=True)
    m = ev['fall'].mean()
    # (1) joint bootstrap
    bs = joint_boot(ev, D, ug)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    r = dict(sample=lab, check='joint bootstrap (events + placebo plants)',
             est=m, lo=lo, hi=hi, p0=ptwo(bs))
    if ug:
        r['p_below_1.1'] = (bs >= 1.1).mean()
    rows.append(r)
    print(f'(1) mean {m:.4f}; joint bootstrap 95% [{lo:.4f}, {hi:.4f}], '
          f'p(0) {r["p0"]:.4f}' + (f', p(mean >= 1.1) {r["p_below_1.1"]:.4f}'
                                   if ug else ''))
    # (2) dependence structures
    sites = [near_sites(P, x.lat, x.lon, int(x.event_year), rad)
             for x in ev.itertuples()]
    ev = ev.assign(stateyear=ev['state'] + '_' + ev['event_year'].astype(str),
                   comp=components(ev, sites))
    for cl in ['fid', 'stateyear', 'state', 'comp']:
        se, Gc = cr_se(ev['fall'], ev[cl])
        p = wild_p(ev['fall'].values, ev[cl].values, 0.0)[1]
        r = dict(sample=lab, check=f'clusters: {cl}', est=m, se=se,
                 clusters=Gc, lo=m - 1.96 * se, hi=m + 1.96 * se, p0=p)
        if ug:
            r['p_below_1.1'] = wild_p(ev['fall'].values, ev[cl].values, 1.1,
                                      True)[1]
        rows.append(r)
        print(f'(2) clusters {cl:9s} ({Gc:2d}): SE {se:.4f}, 95% '
              f'[{m - 1.96 * se:.4f}, {m + 1.96 * se:.4f}], wild p(0) {p:.4f}'
              + (f', wild p(< 1.1) {r["p_below_1.1"]:.4f}' if ug else ''))
    # (3) leave one state out
    lo_rows = []
    for s_ in sorted(ev['state'].unique()):
        d = ev[ev['state'] != s_]
        se, _ = cr_se(d['fall'], d['fid'])
        lo_rows.append(dict(state=s_, dropped=int((ev.state == s_).sum()),
                            est=d['fall'].mean(), se=se))
    L = pd.DataFrame(lo_rows)
    rows.append(dict(sample=lab, check='leave one state out', est=m,
                     lo=L['est'].min(), hi=L['est'].max(),
                     note=f"min when dropping {L.loc[L.est.idxmin(), 'state']},"
                          f" max when dropping {L.loc[L.est.idxmax(), 'state']}"))
    print(f'(3) leave one state out: estimate ranges {L.est.min():.4f} to '
          f'{L.est.max():.4f} (largest |t| change: dropping '
          f"{L.loc[(L.est / L.se - m / cr_se(ev['fall'], ev['fid'])[0]).abs().idxmax(), 'state']})")
    # (4) unadjusted
    sd = pd.read_csv(f"{DIR}/final_{'pm25_35km' if ug else 'so4_100km'}_placebo.csv")
    sdv = (sd['d'] * sd['pre_level']).std() if ug else sd['d'].std()
    mu = ev['fall_raw'].mean(); se = clustered_se(sdv, ev['fid'].values)
    rows.append(dict(sample=lab, check='unadjusted (no control adjustment)',
                     est=mu, lo=mu - 1.96 * se, hi=mu + 1.96 * se,
                     p0=2 * (1 - Phi(abs(mu / se)))))
    print(f'(4) unadjusted mean {mu:.4f}, placebo 95% [{mu - 1.96 * se:.4f}, '
          f'{mu + 1.96 * se:.4f}], p(0) {2 * (1 - Phi(abs(mu / se))):.4f}')

# ---------- (5) what drives the control adjustment ----------
print('\n' + '=' * 70 + '\n(5) Control adjustment: placebo changes vs the '
      "placebo plant's own SO2 change\n" + '=' * 70)
a2 = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                 usecols=['facilityid', 'year', 'so2mass'])
a2['so2mass'] = pd.to_numeric(a2['so2mass'], errors='coerce').fillna(0)
ps = a2.groupby(['facilityid', 'year'])['so2mass'].sum()
for lab, D, ug in [('PM2.5', Dpm, True), ('sulfate', Dso, False)]:
    D = D.copy()
    D['drop_kt'] = [(ps.get((f, y - 1), 0) - ps.get((f, y + 1), 0)) / 1e3
                    for f, y in zip(D['fid'], D['year'])]
    D['y'] = D['d'] * D['lv'] if ug else D['d']
    X = np.column_stack([np.ones(len(D)), D['drop_kt']])
    b = np.linalg.lstsq(X, D['y'], rcond=None)[0]
    e = D['y'] - X @ b
    gi = pd.factorize(D['fid'])[0]; Gc = gi.max() + 1
    Sg = np.stack([np.bincount(gi, X[:, j] * e, minlength=Gc) for j in range(2)], 1)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (Sg.T @ Sg) @ XtXi * Gc / (Gc - 1)
    expl = b[1] * D['drop_kt'].mean()
    rows.append(dict(sample=lab, check='placebo change on own SO2 drop',
                     est=b[1], se=np.sqrt(V[1, 1]),
                     note=f'intercept {b[0]:.4f}; mean drop '
                          f'{D.drop_kt.mean():.2f} kt explains {expl:.4f} of '
                          f'the mean change {D.y.mean():.4f}'))
    print(f'  {lab}: slope {b[1]:.4f} per 1,000 t (cluster SE '
          f'{np.sqrt(V[1, 1]):.4f}); placebo plants cut SO2 by '
          f'{D.drop_kt.mean():.2f} kt on average between y-1 and y+1, which '
          f'accounts for {expl:.4f} of the mean change {D.y.mean():.4f} '
          f'({"ug/m3" if ug else "log points"})')

# ---------- (6) H3a and H3d(b) without the H3e failures ----------
C = pd.read_csv(f'{OUT}/events_so2_check.csv')
fails = set(zip(C.loc[C['so2_dlog'] <= 0, 'fid'], C.loc[C['so2_dlog'] <= 0,
                                                        'event_year']))
print(f'\n(6) H3e failures dropped: {sorted(fails)}')
for lab, ev, x in [('H3a PM2.5 slope', F, 'pred_main'),
                   ('H3d(b) sulfate slope', SV, 'pred_rel_main')]:
    d = ev[[k not in fails for k in zip(ev['fid'], ev['event_year'])]]
    b = slope0(d[x], d['fall'])
    bs = cluster_boot(d, lambda z: slope0(z[x], z['fall']))
    rows.append(dict(sample=lab, check='without H3e failures', est=b,
                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     p_slope1=boot_slope_p(bs), events=len(d)))
    print(f'  {lab}: {len(d)} events, slope {b:.2f} '
          f'[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}], '
          f'p(slope = 1) {boot_slope_p(bs):.4f}')

# ---------- (7) Q2 with the low release layer ----------
print('\n(7) Q2 damages by release layer (InMAP marginal mortality)')
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
G_ = np.load(f'{DIR}/isrm_grid.npz')        # same reordering as valuepred.py
if not (np.allclose(mv['Location_W'].values, G_['W'])
        and np.allclose(mv['Location_S'].values, G_['S'])):
    key = pd.Series(range(len(mv)), index=pd.MultiIndex.from_arrays(
        [mv['Location_W'].round(0), mv['Location_S'].round(0)]))
    idx = key.reindex(pd.MultiIndex.from_arrays(
        [np.round(G_['W'], 0), np.round(G_['S'], 0)])).values
    assert not np.isnan(idx).any(), 'grid mismatch'
    mv = mv.iloc[idx.astype(int)].reset_index(drop=True)
Ud = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
Ud = Ud[Ud['cell'] >= 0]
CF = pd.read_csv(f'{DIR}/counterfactual_paths_v2.csv')[['uid', 'ratio_so2',
                                                        'ratio_nox']]
ST2T = 0.90718474
for h in ['high', 'low']:
    ms = mv[f'MM_SO2_{h}'].values[Ud['cell'].astype(int)] * ST2T
    mn = mv[f'MM_NOx_{h}'].values[Ud['cell'].astype(int)] * ST2T
    # deaths (MM) for the shares, multiples and counterfactual, as in
    # outcomes4.py; dollar damages (MD), as in valuepred.py, printed alongside
    ds = mv[f'MD_SO2_{h}'].values[Ud['cell'].astype(int)]
    dn = mv[f'MD_NOx_{h}'].values[Ud['cell'].astype(int)]
    for base in ['t10', 'peak']:
        bd_ = Ud[f'so2_{base}'] * ds + Ud[f'nox_{base}'] * dn
        fd_ = Ud['so2_final'] * ds + Ud['nox_final'] * dn
        k_ = bd_ > 0
        print(f"  {h}, {base}, dollar damages: share before last full year "
              f"{100 * (1 - fd_[k_].sum() / bd_[k_].sum()):.1f}%, multiple "
              f"{bd_[k_].sum() / fd_[k_].sum():.2f}")
    fin = Ud['so2_final'] * ms + Ud['nox_final'] * mn
    out = {}
    for base in ['t10', 'peak']:
        bd = Ud[f'so2_{base}'] * ms + Ud[f'nox_{base}'] * mn
        k = bd > 0
        out[base] = (1 - fin[k].sum() / bd[k].sum(), bd[k].sum() / fin[k].sum())
    M = Ud.assign(fin=fin, t10=Ud['so2_t10'] * ms + Ud['nox_t10'] * mn,
                  ms=ms, mn=mn).merge(CF, on='uid')
    cfd = (M.so2_final * M.ratio_so2 * M.ms + M.nox_final * M.ratio_nox * M.mn).sum()
    rows.append(dict(sample=f'Q2 {h} release layer', check='damage shares and '
                     'multiples', est=out['t10'][1],
                     note=f"share before last full year: t10 {out['t10'][0]:.3f}, "
                          f"peak {out['peak'][0]:.3f}; multiples t10 "
                          f"{out['t10'][1]:.2f}, peak {out['peak'][1]:.2f}; "
                          f"counterfactual ({len(M)} exits): last-full-year "
                          f"{M.fin.sum() / cfd:.2f}x, ten-years-before "
                          f"{M.t10.sum() / cfd:.2f}x"))
    print(f"  {h}: shares {100 * out['t10'][0]:.0f}% (t10) and "
          f"{100 * out['peak'][0]:.0f}% (peak); multiples {out['t10'][1]:.2f} "
          f"and {out['peak'][1]:.2f}; counterfactual {M.fin.sum() / cfd:.2f}x "
          f"and {M.t10.sum() / cfd:.2f}x")

save(pd.DataFrame(rows), 'robustness_batchC.csv')
print('\nSHA-256:')
for p in saved:
    print(f'  {os.path.basename(p)}  {hashlib.sha256(open(p, "rb").read()).hexdigest()}')
print('\nDone. Send this output.')
