# POST-REGISTRATION outcome analysis, part 3: H3f (satellite aerosol, MAIAC)
# exactly as pre-registered (PDF section 5, H3f; section 6 inference).
# Run after all maiac_events_v2_* exports have COMPLETED.
#  Plant-year statistic: mean over days of (downwind - upwind) sector AOD,
#  sectors with 20+ valid 1 km pixels, 20+ such days a year.
#  Event fall: year before minus year after (event year excluded).
#  Sample: SO2 drop >= 5,000 t, events 2005-2018 (109 at filing).
#  Expected fall: SO2 drop x control slope (never-exit plants >= 60 km
#  from exits; 0.000083 AOD per 1,000 t at filing).
#  H3f(a): mean fall vs 0 (two-sided); H3f(b): slope of falls on expected
#  falls vs 1, calibration uncertainty carried by a plant bootstrap of the
#  control slope. Inference as for the ground tests; most conservative p is
#  primary. Then Holm across the five confirmatory tests available (H4 in
#  2027).
DIR = '/content/drive/MyDrive/coal_closures'
src = open(f'{DIR}/outcomes1.py').read().split('results, events_out')[0]
exec(src)
import glob

more = ['maiac_control_plants.csv', 'maiac_control_plantyears.csv',
        'maiac_placebo_draws.csv', 'wind_daily_plants.csv']
bad = [f for f in more if hashlib.sha256(open(f'{DIR}/{f}', 'rb').read())
       .hexdigest() != fpr.get(f)]
if bad:
    raise SystemExit(f'STOP: these files differ from the registered '
                     f'fingerprints: {bad}')
print('MAIAC inputs also match the registered fingerprints.')

files = sorted(glob.glob('/content/drive/MyDrive/maiac_exports/'
                         'maiac_events_v2_*.csv'))
print('Event export files found:', len(files), '(expected 32)')
if len(files) < 32:
    raise SystemExit('Wait until all 32 exports have COMPLETED.')
m = pd.concat([pd.read_csv(f) for f in files])
m = m[m['count'] >= 20]
w = pd.read_csv(f'{DIR}/wind_daily_plants.csv')
w['to'] = (np.degrees(np.arctan2(w['u'], w['v'])) + 360) % 360
w['down'] = (((w['to'] + 22.5) % 360) // 45).astype(int)
w['up'] = (w['down'] + 4) % 8
x = m.merge(w[['facilityid', 'date', 'down', 'up']],
            left_on=['fid', 'date'], right_on=['facilityid', 'date'])
dn = x[x['sector'] == x['down']][['fid', 'date', 'mean']]
upx = x[x['sector'] == x['up']][['fid', 'date', 'mean']]
d = dn.merge(upx, on=['fid', 'date'], suffixes=('_d', '_u'))
d['du'] = d['mean_d'] - d['mean_u']
d['yr'] = d['date'].str[:4].astype(int)
py = d.groupby(['fid', 'yr']).agg(du=('du', 'mean'), n=('du', 'size'))
py = py[py['n'] >= 20]['du']
print(f'Exit-plant years with 20+ valid day pairs: {len(py)} '
      f'({py.index.get_level_values(0).nunique()} plants)')

# control calibration and placebo (registered files)
C = pd.read_csv(f'{DIR}/maiac_control_plantyears.csv')
def cslope(df):
    X = np.column_stack([np.ones(len(df)), df['so2'] / 1000])
    return np.linalg.lstsq(X, df['du'].values, rcond=None)[0][1]
b_cal = cslope(C)
cg = C.groupby('fid').indices; ck = list(cg)
Dc = []
for f, g in C.groupby('fid'):
    g = g.set_index('yr')['du']
    for y in range(2005, 2019):
        if y - 1 in g.index and y + 1 in g.index:
            Dc.append((y, g[y - 1] - g[y + 1]))
Dc = pd.DataFrame(Dc, columns=['year', 'd'])
ym, pool = Dc.groupby('year')['d'].mean().to_dict(), Dc['d'].mean()
sd_m = pd.read_csv(f'{DIR}/maiac_placebo_draws.csv')['d'].std()
print(f'Control slope {b_cal:.6f} AOD per 1,000 t (registered '
      f'0.000083); placebo SD {sd_m:.4f}; placebo mean {pool:.5f}')

# events
ev = E[(E['so2_drop'] >= 5000) & E['event_year'].between(2005, 2018)].copy()
def fall(f, y):
    if (f, y - 1) in py.index and (f, y + 1) in py.index:
        return py[(f, y - 1)] - py[(f, y + 1)]
    return np.nan
ev['fall_raw'] = [fall(r.fid, r.event_year) for r in ev.itertuples()]
n0 = len(ev)
ev = ev.dropna(subset=['fall_raw']).copy()
print(f'H3f sample: {len(ev)} of {n0} events have valid plant-years on both '
      'sides')
ev['fall'] = ev['fall_raw'] - ev['event_year'].map(ym).fillna(pool)
ev['expected'] = b_cal * ev['so2_drop'] / 1000

# placebo dates at the exit plants (years >= 3 from any exit there)
cand = {}
for f in ev['fid'].unique():
    ys = exit_years.get(int(f), set())
    cand[f] = np.array([fall(f, y) for y in range(2005, 2019)
                        if all(abs(y - e) >= 3 for e in ys)
                        and np.isfinite(fall(f, y))])
use = ev[ev['fid'].map(lambda f: len(cand[f]) > 0)]
draws = np.array([np.mean([rng.choice(cand[f]) for f in use['fid']])
                  for _ in range(B)])
r = summarise('H3f(a)  MAIAC down-minus-up AOD fall', ev['fall'].values,
              ev['fid'].values, sd_m, None, use,
              use['fall_raw'].mean(), draws)
r['mean_expected'] = ev['expected'].mean()
show(r)
print(f"  mean expected fall {r['mean_expected']:.5f} AOD; unadjusted mean "
      f"{ev['fall_raw'].mean():.5f}")

# H3f(b): slope vs 1 with calibration uncertainty
gi = ev.groupby('fid').indices; gk = list(gi)
bs = np.empty(B)
for i in range(B):
    cp = rng.choice(len(ck), len(ck))
    bc = cslope(C.iloc[np.concatenate([cg[ck[k]] for k in cp])])
    ep = rng.choice(len(gk), len(gk))
    e_ = ev.iloc[np.concatenate([gi[gk[k]] for k in ep])]
    bs[i] = slope0(bc * e_['so2_drop'] / 1000, e_['fall'])
b = slope0(ev['expected'], ev['fall'])
rb = dict(test='H3f(b)  MAIAC slope on expected fall', events=len(ev),
          slope=b, ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
          p_slope_eq_1=boot_slope_p(bs))
print(f"\nH3f(b): slope {b:.2f}, 95% CI (events and calibration "
      f"bootstrapped) [{rb['ci'][0]:.2f}, {rb['ci'][1]:.2f}], p(slope = 1) "
      f"{rb['p_slope_eq_1']:.4f}")
ev.to_csv(f'{OUT}/events_maiac.csv', index=False)
pd.DataFrame([r, rb]).to_csv(f'{OUT}/results_part3_maiac.csv', index=False)

# Holm across the confirmatory family available now (H4 pending, 2027)
R1 = pd.read_csv(f'{OUT}/results_part1.csv')
def g(test, col):
    return float(R1.loc[R1['test'].str.startswith(test), col].iloc[0])
fam = {'H3b': max(g('H3b/H3c  total', 'p_alt_primary'), 1 / 9999),
       'H3c': g('H3b/H3c  total', 'p0_primary'),
       'H3d(a)': g('H3d(a)  sulfate, 100', 'p0_primary'),
       'H3d(b)': g('H3d(b)  sulfate slope, 100', 'p_slope_eq_1'),
       'H3f(a)': max(r['p0_primary'], 1 / 9999)}
for k_tot, note in [(6, 'conservative bound: six tests, H4 assumed to have the largest p'),
                    (5, 'the five tests available now; indicative only')]:
    print(f'\nHolm-Bonferroni at 5% ({note}):')
    alive = True
    for i, (k, p) in enumerate(sorted(fam.items(), key=lambda t: t[1])):
        thr = 0.05 / (k_tot - i)
        alive = alive and p <= thr
        print(f'  {k:7s} p {p:.4f}  threshold {thr:.4f}  '
              f'{"REJECTED" if alive else "not rejected"}')
print('\nSaved events_maiac.csv and results_part3_maiac.csv to', OUT)
