# POST-REGISTRATION, part 4 (no new outcome tests):
#  (A) Q2 counterfactual, corrected. The filed version took each donor's
#      later/earlier SO2 ratio with the year-before SO2 clipped at 1 ton,
#      so donors with almost no SO2 gave ratios up to 386. Here the donor
#      ratio is a ratio of donor TOTALS (sum of later emissions / sum of
#      year-before emissions over the 5 matched units), and covariate
#      balance before exit is reported. Same matching as steps345.py.
#  (B) Sulfate fall in µg/m³ against the predicted absolute fall
#      (exploratory: converts H3d to the units used for valuation).
#  (C) The 51 primary PM2.5 events as an appendix table (markdown).
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd

# ---------- (A) counterfactual, ratio of donor totals ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'unitid', 'year', 'statecode',
                         'so2mass', 'noxmass', 'heatinput',
                         'primaryfuelinfo'])
for c in ['so2mass', 'noxmass', 'heatinput']:
    a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
a['uid'] = a['facilityid'].astype(str) + '_' + a['unitid'].astype(str)
a['coal'] = a['primaryfuelinfo'].fillna('').str.contains('Coal') \
    & (a['heatinput'] > 0)
coal_uids = set(a.loc[a['coal'], 'uid'])
A = a[a['uid'].isin(coal_uids)].pivot_table(
    index='uid', columns='year', values=['so2mass', 'noxmass', 'heatinput'],
    aggfunc='sum').fillna(0)
state = a.drop_duplicates('uid').set_index('uid')['statecode']
lastcoal = a[a['coal']].groupby('uid')['year'].max()
U = pd.read_csv(f'{DIR}/us_unit_exits_damages.csv')
mv = pd.read_csv(f'{DIR}/isrm_marginal_values.csv')
ST2T = 0.90718474
H = A['heatinput']; S2 = A['so2mass']; NX = A['noxmass']
rows, bal, paths = [], [], []
for u in U.itertuples():
    T = u.last_coal_year
    if T - 5 < 1997 or T + 5 > 2025:
        continue
    pool = [v for v in lastcoal.index[lastcoal >= T + 5]
            if v != u.uid and (H.loc[v, T - 5:T - 1] > 0).all()]
    if len(pool) < 10:
        continue
    same = [v for v in pool if state.get(v) == state.get(u.uid)]
    pool = same if len(same) >= 5 else pool
    h1, h5 = H.at[u.uid, T - 1], H.at[u.uid, T - 5]
    if h1 <= 0 or h5 <= 0:
        continue
    x0 = np.array([np.log(h1 / h5), np.log(h1),
                   np.log((S2.at[u.uid, T - 1] + 1) / h1)])
    X = np.column_stack([np.log(H.loc[pool, T - 1] / H.loc[pool, T - 5]),
                         np.log(H.loc[pool, T - 1]),
                         np.log((S2.loc[pool, T - 1] + 1)
                                / H.loc[pool, T - 1])])
    sdv = X.std(0) + 1e-9
    dist = np.sqrt((((X - x0) / sdv) ** 2).sum(1))
    nn = [pool[i] for i in np.argsort(dist)[:5]]
    yrs5 = list(range(T + 1, T + 6))
    s_pre, n_pre = S2.loc[nn, T - 1].sum(), NX.loc[nn, T - 1].sum()
    r_s = S2.loc[nn, yrs5].mean(axis=1).sum() / s_pre if s_pre > 0 else 1.0
    r_n = NX.loc[nn, yrs5].mean(axis=1).sum() / n_pre if n_pre > 0 else 1.0
    c = int(u.cell)
    mm_s, mm_n = mv['MM_SO2_high'].values[c], mv['MM_NOx_high'].values[c]
    rows.append(dict(uid=u.uid, T=T, ratio_so2=r_s, ratio_nox=r_n,
                     deaths_final=(u.so2_final * mm_s + u.nox_final * mm_n)
                     * ST2T,
                     deaths_cf=(u.so2_final * r_s * mm_s
                                + u.nox_final * r_n * mm_n) * ST2T,
                     deaths_t10=(u.so2_t10 * mm_s + u.nox_t10 * mm_n) * ST2T))
    # event-time paths k = -5..+5, each relative to t-1 (for the figure)
    for k in range(-5, 6):
        paths.append(dict(uid=u.uid, k=k, w=S2.at[u.uid, T - 1],
                          s_exit=S2.at[u.uid, T + k],
                          h_exit=H.at[u.uid, T + k], h_w=h1,
                          s_donor=S2.loc[nn, T + k].sum() / s_pre
                          if s_pre > 0 else np.nan,
                          h_donor=H.loc[nn, T + k].sum()
                          / H.loc[nn, T - 1].sum()))
    # balance: pre-exit trends of the exiting unit and its donors
    for k in range(5, 0, -1):
        bal.append(dict(uid=u.uid, k=-k,
                        h_exit=H.at[u.uid, T - k] / h1,
                        h_donor=(H.loc[nn, T - k] / H.loc[nn, T - 1]).median(),
                        s_exit=S2.at[u.uid, T - k] / max(S2.at[u.uid, T - 1], 1),
                        s_donor=(S2.loc[nn, T - k].sum()
                                 / max(S2.loc[nn, T - 1].sum(), 1))))
CF = pd.DataFrame(rows)
CF.to_csv(f'{DIR}/counterfactual_paths_v2.csv', index=False)
print(f'(A) Counterfactual, ratio of donor totals: {len(CF)} exits')
print('  donor SO2 ratio: median %.2f, p95 %.2f, max %.2f; NOx median %.2f'
      % (CF.ratio_so2.median(), CF.ratio_so2.quantile(.95),
         CF.ratio_so2.max(), CF.ratio_nox.median()))
f, cf, t10 = CF.deaths_final.sum(), CF.deaths_cf.sum(), CF.deaths_t10.sum()
print(f'  deaths/yr avoided: counterfactual {cf:,.0f}; last-full-year rule '
      f'{f:,.0f} ({f / cf:.2f}x); ten-years-before rule {t10:,.0f} '
      f'({t10 / cf:.2f}x)')
top = CF.nlargest(2, 'deaths_cf')
print(f'  largest two units supply {top.deaths_cf.sum() / cf:.0%} of '
      'counterfactual deaths:', ', '.join(top.uid))
B = pd.DataFrame(bal).drop(columns='uid').groupby('k').median()
print('  Balance, medians across exits (each series relative to t-1 = 1):')
print(B.round(3).to_string())
P = pd.DataFrame(paths)
P = P[P['s_donor'].notna()]
G = P.groupby('k').apply(lambda d: pd.Series(dict(
    exit_so2=d.s_exit.sum() / d.w.sum(),
    donor_so2=(d.s_donor * d.w).sum() / d.w.sum(),
    exit_heat=d.h_exit.sum() / d.h_w.sum(),
    donor_heat=(d.h_donor * d.h_w).sum() / d.h_w.sum())))
G.to_csv(f'{DIR}/outcomes/cf_event_paths.csv')
print('  Event-time paths, tonnage-weighted, t-1 = 1 (figure data, saved '
      'to outcomes/cf_event_paths.csv):')
print(G.round(3).to_string())

# ---------- (B) sulfate fall in µg/m³ ----------
S = pd.read_csv(f'{DIR}/outcomes/events_so4_100km.csv')
S['fall_ug'] = S['fall'] * S['pre_level']
print(f'\n(B) Sulfate, {len(S)} events (exploratory): observed fall '
      f'{S.fall_ug.mean():.3f} µg/m³ (control-adjusted; near-monitor '
      f'level {S.pre_level.mean():.2f}) vs predicted {S.so4_main.mean():.3f}'
      f' µg/m³; ratio {S.fall_ug.mean() / S.so4_main.mean():.2f}')
pl = pd.read_csv(f'{DIR}/final_so4_100km_placebo.csv')['d']
se = pl.std() * S['pre_level'].mean() / np.sqrt(len(S))
print(f'  approximate placebo SE {se:.3f} µg/m³; 95% interval '
      f'[{S.fall_ug.mean() - 1.96 * se:.3f}, '
      f'{S.fall_ug.mean() + 1.96 * se:.3f}]')

# ---------- (C) event table ----------
F = pd.read_csv(f'{DIR}/outcomes/events_pm25_35km.csv').sort_values(
    ['event_year', 'name'])
print('\n(C) Appendix table (copy as is):\n')
print('| Plant | Exit year | SO2 drop (short tons) | Release layer | '
      'Monitors | Prediction (µg/m³) | Observed fall, adjusted (µg/m³) |')
print('| --- | --- | --- | --- | --- | --- | --- |')
for r in F.itertuples():
    print(f'| {r.name} | {r.event_year} | {r.so2_drop:,.0f} | '
          f'{"low" if r.layer == 1 else "high"} | {int(r.n_obs)} | '
          f'{r.pred_main:.3f} | {r.fall:.3f} |')
