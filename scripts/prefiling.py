# Pre-filing step: (1) placebo SD on 2004-2018 never-exit plants;
# (2) SHA-256 fingerprints of every frozen file and script.
# Run AFTER isrmpredict.py has regenerated model_predictions_us.csv.
# Reads PM2.5 values only at never-exit control plants, using only monitors
# more than 25 km from every plant with a coal exit (any year).
DIR = '/content/drive/MyDrive/coal_closures'
import os, glob, hashlib
import numpy as np, pandas as pd

# ---------- (1) placebo SD ----------
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
a['coal'] = (a['primaryfuelinfo'].fillna('').str.contains('Coal')
             & (a['heatinput'] > 0))
cy = a[a['coal']].groupby('facilityid')['year'].apply(set)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ctrl_fids = [f for f, ys in cy.items()
             if set(range(2003, 2020)) <= ys and f not in exit_fids]
fac = pd.read_csv(f'{DIR}/us_epa_facilities.csv').set_index('facilityid')
coal_fac = fac.loc[fac.index.intersection(cy.index)]
ctrl = fac.loc[fac.index.intersection(ctrl_fids)]
exits = fac.loc[fac.index.intersection(list(exit_fids))]
print(f'Never-exit control plants (coal every year 2003-2019, no unit '
      f'exit 2005-2024): {len(ctrl)}')

frames = [pd.read_csv(f'{DIR}/aqs/pm25_monthly_{y}.csv')
          for y in range(2002, 2020)]
q = pd.concat(frames)
q = q[(q['ndays'] >= 8) & (q['pm'] > 0)]
q['yr'] = q['ym'] // 100
sites = q.groupby('site')[['lat', 'lon', 'state']].first()

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
Sx = sites[['lat', 'lon']].values
def min_dist(pts):
    m = np.full(len(Sx), np.inf)
    for la, lo in pts:
        m = np.minimum(m, km(Sx[:, 0], Sx[:, 1], la, lo))
    return m
sites['d_coal'] = min_dist(coal_fac[['latitude', 'longitude']].values)
sites['d_exit'] = min_dist(exits[['latitude', 'longitude']].values)
bg_sites = set(sites.index[sites['d_coal'] > 50])
allowed = set(sites.index[sites['d_exit'] > 25])

q['lpm'] = np.log(q['pm'])
bg = (q[q['site'].isin(bg_sites)].groupby(['state', 'ym'])['lpm'].mean()
      .rename('bg'))
q = q.join(bg, on=['state', 'ym']).dropna(subset=['bg'])
q['rel'] = q['lpm'] - q['bg']
ann = q.groupby(['site', 'yr']).agg(rel=('rel', 'mean'), n=('rel', 'size'),
                                    pm=('pm', 'mean'))
ann = ann[ann['n'] >= 8]

# event years of the primary sample (years only)
P = pd.read_csv(f'{DIR}/model_predictions_us_events.csv')
prim = P[(P['n_near_bg'] > 0) & (P['so2_drop'] >= 5000)].copy()
P2 = pd.read_csv(f'{DIR}/model_predictions_us_events_v2.csv')
p2 = P2[(P2['n_near_bg'] > 0) & (P2['so2_drop'] >= 5000)]
prim['pred_contrast_high'] = p2['pred_main'].mean()   # main prediction
prim['pred_contrast_low'] = p2['pred_low'].mean()
years = prim['event_year'].values
print('Primary-sample events:', len(prim), ' event years:',
      dict(zip(*np.unique(years, return_counts=True))))

# change for each control plant and each candidate event year
D, LV = {}, {}
for fid, r in ctrl.iterrows():
    d = km(Sx[:, 0], Sx[:, 1], r['latitude'], r['longitude'])
    near = [s for s in sites.index[d <= 25] if s in allowed]
    for y in range(2004, 2019):
        vals, lv = [], []
        for s in near:
            if (s, y - 1) in ann.index and (s, y + 1) in ann.index:
                vals.append(ann.at[(s, y - 1), 'rel']
                            - ann.at[(s, y + 1), 'rel'])
                lv.append(ann.at[(s, y - 1), 'pm'])
        if vals:
            D[(fid, y)], LV[(fid, y)] = np.mean(vals), np.mean(lv)
plants_ok = sorted({f for f, _ in D})
print('Control plants with an allowed monitor and coverage:',
      len(plants_ok))

rng = np.random.default_rng(20261109)
draws, levels = [], []
for b in range(1000):
    for f in plants_ok:
        y = int(rng.choice(years))
        if (f, y) in D:
            draws.append(D[(f, y)]); levels.append(LV[(f, y)])
draws, levels = np.array(draws), np.array(levels)
sd = draws.std()
lvl = levels.mean()
se = sd / np.sqrt(len(prim))
print(f'\nPlacebo change, log points: mean {draws.mean():.4f}, '
      f'SD {sd:.4f} ({len(draws):,} plant-draws)')
print(f'Pre-year PM2.5 at these monitors: {lvl:.1f} µg/m³')
print(f'Implied SE for {len(prim)} events: {se:.4f} log points '
      f'= {se * lvl:.3f} µg/m³; MDE (80% power) = {2.8 * se * lvl:.2f} '
      f'µg/m³')
print(f'Distance from 1.1 µg/m³ to the mean prediction (main '
      f'{prim.pred_contrast_high.mean():.3f}): '
      f'{(1.1 - prim.pred_contrast_high.mean()) / (se * lvl):.1f} SE; '
      f'(low {prim.pred_contrast_low.mean():.3f}): '
      f'{(1.1 - prim.pred_contrast_low.mean()) / (se * lvl):.1f} SE')
pd.DataFrame({'placebo_change_log': draws, 'pre_level': levels}).to_csv(
    f'{DIR}/placebo_draws_2004_2018.csv', index=False)

# ---------- (1b) SO2 check coverage for the final primary sample ----------
F = pd.read_csv(f'{DIR}/final_pm25_35km_predictions.csv')
F = F[(F['n_near_bg'] > 0) & (F['so2_drop'] >= 5000)]
Ev = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
F = F.merge(Ev[['fid', 'event_year', 'lat', 'lon']], on=['fid', 'event_year'])
s2 = pd.concat([pd.read_csv(f'{DIR}/so2/so2_monthly_{y}.csv',
                            usecols=['site', 'ym', 'lat', 'lon', 'ndays'])
                for y in range(2004, 2025)])
s2 = s2[s2['ndays'] >= 8]; s2['yr'] = s2['ym'] // 100
ss = s2.groupby('site')[['lat', 'lon']].first()
ny = s2.groupby(['site', 'yr']).size()
sy = ny[ny >= 8].reset_index().groupby('site')['yr'].apply(set)
F['so2_monitors_25km'] = [
    sum(1 for s_ in ss.index[km(ss['lat'].values, ss['lon'].values,
                                r.lat, r.lon) <= 25]
        if {r.event_year - 1, r.event_year + 1} <= sy.get(s_, set()))
    for r in F.itertuples()]
F[['fid', 'name', 'event_year', 'so2_monitors_25km']].to_csv(
    f'{DIR}/so2_check_final.csv', index=False)
print(f'SO2 check: {(F.so2_monitors_25km > 0).sum()} of {len(F)} final '
      'primary events have an SO2 monitor within 25 km')

# ---------- (1c) MAIAC calibration uncertainty (control plants only) ----
import glob
files_m = sorted(glob.glob('/content/drive/MyDrive/maiac_exports/'
                           'maiac_controls_*.csv'))
mm = pd.concat([pd.read_csv(f) for f in files_m])
mm = mm[mm['count'] >= 20]
wd = pd.read_csv(f'{DIR}/wind_daily_plants.csv')
wd['to'] = (np.degrees(np.arctan2(wd['u'], wd['v'])) + 360) % 360
wd['down'] = (((wd['to'] + 22.5) % 360) // 45).astype(int)
wd['up'] = (wd['down'] + 4) % 8
xm = mm.merge(wd[['facilityid', 'date', 'down', 'up']],
              left_on=['fid', 'date'], right_on=['facilityid', 'date'])
dn = xm[xm['sector'] == xm['down']][['fid', 'date', 'mean']]
upm = xm[xm['sector'] == xm['up']][['fid', 'date', 'mean']]
dd = dn.merge(upm, on=['fid', 'date'], suffixes=('_d', '_u'))
dd['du'] = dd['mean_d'] - dd['mean_u']
dd['yr'] = dd['date'].str[:4].astype(int)
pym = dd.groupby(['fid', 'yr']).agg(du=('du', 'mean'), n=('du', 'size')) \
        .reset_index()
pym = pym[pym['n'] >= 20]
am = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                 usecols=['facilityid', 'year', 'so2mass'])
am['so2mass'] = pd.to_numeric(am['so2mass'], errors='coerce').fillna(0)
pym = pym.join(am.groupby(['facilityid', 'year'])['so2mass'].sum()
               .rename('so2'), on=['fid', 'yr']).dropna()
pym.to_csv(f'{DIR}/maiac_control_plantyears.csv', index=False)
def slope(df):
    X = np.column_stack([np.ones(len(df)), df['so2'] / 1000])
    return np.linalg.lstsq(X, df['du'].values, rcond=None)[0][1]
b0 = slope(pym)
rngb = np.random.default_rng(20261109)
fids = pym['fid'].unique()
bs = []
for _ in range(1000):
    pick = rngb.choice(fids, len(fids))
    bs.append(slope(pd.concat([pym[pym['fid'] == f] for f in pick])))
bs = np.array(bs)
print(f'MAIAC calibration slope {b0:.6f} AOD per 1,000 t SO2; plant '
      f'bootstrap SE {bs.std():.6f}; 90% interval '
      f'[{np.percentile(bs, 5):.6f}, {np.percentile(bs, 95):.6f}]; '
      f'share of draws > 0: {(bs > 0).mean():.2f}')

# ---------- (2) fingerprints ----------
files = ['model_predictions_us_events_v2.csv',
         'model_predictions_us_events.csv', 'model_predictions_us.csv',
         'us_epa_annual_all.csv', 'step1/nei_pm25_by_oris_v3.csv',
         'step1/eia860_stacks.csv',
         'model_predictions_us_monitors.csv', 'us_plant_exit_events.csv',
         'us_unit_exits.csv', 'us_unit_exits_damages.csv',
         'us_unit_exit_trajectories.csv', 'us_closure_dates.csv',
         'eu_unit_exits.csv', 'eu_coal_sites.csv',
         'isrm_coal_facility_cells.csv', 'placebo_draws_2004_2018.csv',
         'so4_event_coverage.csv', 'so4_primary_events.csv',
         'so4_placebo_draws.csv', 'so2_check_coverage.csv',
         'counterfactual_paths.csv', 'downwind_placebo_draws.csv',
         'wind_daily_plants.csv', 'final_pm25_35km_predictions.csv',
         'final_pm25_35km_placebo.csv', 'final_so4_100km_predictions.csv',
         'final_so4_100km_placebo.csv', 'final_so4_50km_predictions.csv',
         'final_so4_50km_placebo.csv', 'design_tuning.csv',
         'hourly_event_coverage.csv', 'downwind_hourly_placebo_draws.csv',
         'aqs_hourly_monitor_years.csv', 'maiac_control_plants.csv',
         'maiac_placebo_draws.csv', 'so2_check_final.csv',
         'eu_station_coverage.csv', 'maiac_control_plantyears.csv']
files = [f for f in files if os.path.exists(f'{DIR}/{f}')]
files += sorted(os.path.basename(p) for p in glob.glob(f'{DIR}/*.py'))
rows = []
for f in files:
    p = f'{DIR}/{f}'
    rows.append(dict(file=f, bytes=os.path.getsize(p),
                     sha256=hashlib.sha256(open(p, 'rb').read())
                     .hexdigest()))
fp = pd.DataFrame(rows)
fp.to_csv(f'{DIR}/fingerprints.csv', index=False)
pd.set_option('display.width', 200, 'display.max_colwidth', 80)
print('\nFINGERPRINTS (saved to fingerprints.csv):')
print(fp.to_string(index=False))
