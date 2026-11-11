# POST-REGISTRATION, exploratory: air quality by year around exit.
# For the registered samples (PM2.5 35 km, 51 events; sulfate 100 km,
# 44 events), the near-minus-background log concentration in years
# T-3..T+3 relative to T-1, the same statistic as the registered test
# (k = +1 is its pre-minus-post change with the sign flipped). The same
# path at never-exit plants (pseudo-years 2005-2018) is subtracted, as in
# the registered control adjustment. 95% intervals: cluster bootstrap by
# plant (placebo noise not added). Figure data: outcomes/eventtime_*.csv.
DIR = '/content/drive/MyDrive/coal_closures'
exec(open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0])

K = range(-3, 4)
def path(P, lat, lon, T, radius, need_cell=True):
    st, Sx, ann = P['st'], P['Sx'], P['ann']
    d = km(Sx[:, 0], Sx[:, 1], lat, lon)
    out = {k: [] for k in K}; lv = []
    for s in st.index[d <= radius]:
        if (s, T - 1) not in ann.index:
            continue
        if need_cell and st.at[s, 'cell'] < 0:
            continue
        if not (st['bg'] & (st['state'] == st.at[s, 'state'])).any():
            continue
        base = ann.at[(s, T - 1), 'rel']; lv.append(ann.at[(s, T - 1), 'lv'])
        for k in K:
            if (s, T + k) in ann.index:
                out[k].append(ann.at[(s, T + k), 'rel'] - base)
    return {k: (np.mean(v) if v else np.nan) for k, v in out.items()}, \
        (np.mean(lv) if lv else np.nan)

def run(P, ev, radius, to_ug, label, need_cell=True):
    rows = []
    for r in ev.itertuples():
        p, lv = path(P, r.lat, r.lon, int(r.event_year), radius, need_cell)
        rows.append(dict(fid=r.fid, lv=lv, **{f'k{k}': p[k] for k in K}))
    Ev = pd.DataFrame(rows)
    allowed = set(P['st'].index[P['st']['d_exit'] > radius])
    Pst = dict(P); Pst['st'] = P['st'].loc[sorted(allowed)]
    Pst['Sx'] = Pst['st'][['lat', 'lon']].values
    pl = []
    for c in ctrl.itertuples():
        for y in range(2005, 2019):
            p, _ = path(Pst, c.latitude, c.longitude, y, radius, need_cell)
            pl.append({f'k{k}': p[k] for k in K})
    Pl = pd.DataFrame(pl).mean()
    scale = Ev['lv'] if to_ug else 1.0
    A = pd.DataFrame({f'k{k}': (Ev[f'k{k}'] - Pl[f'k{k}']) * scale for k in K})
    A['fid'] = Ev['fid']
    g = {f: d for f, d in A.groupby('fid')}; keys = list(g)
    bs = np.array([pd.concat([g[keys[i]] for i in rng.integers(
        0, len(keys), len(keys))])[[f'k{k}' for k in K]].mean().values
        for _ in range(1999)])
    est = A[[f'k{k}' for k in K]].mean().values
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    O = pd.DataFrame({'k': list(K), 'estimate': est, 'lo': lo, 'hi': hi,
                      'events': [A[f'k{k}'].notna().sum() for k in K]})
    O.to_csv(f'{OUT}/eventtime_{label}.csv', index=False)
    unit = 'ug/m3' if to_ug else 'log points'
    print(f'\n{label} ({unit}, relative to T-1, control-adjusted):')
    print(O.round(3).to_string(index=False))

q = pd.concat([pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
               for y in range(2000, 2026)])
PM = prep(q, 'pm', 8); del q
F = pd.read_csv(f'{OUT}/events_pm25_35km.csv')
run(PM, F, 35, True, 'pm25_35km')
q = pd.concat([pd.read_csv(f'{DIR}/so4/so4_monthly_{y}.csv')
               for y in range(2004, 2025)])
SO = prep(q, 'so4', 3); del q
Fs = pd.read_csv(f'{OUT}/events_so4_100km.csv')
run(SO, Fs, 100, False, 'so4_100km')
