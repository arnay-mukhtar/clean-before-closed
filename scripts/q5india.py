# Q5, India (pre-registered descriptive question; data first viewed 10 Nov
# 2026, after filing). Version 3. How hard did Indian coal units run before
# they retired?
# What the CEA CO2 Baseline Database allows: for coal stations it reports net
# generation for the STATION in every fiscal year (unit "0" rows); unit rows
# carry generation only for a version's latest year or two. So output is
# measured at station level as the plant load factor (PLF) = net generation
# / (capacity at the start of the year x 8.76), and each retiring unit is
# given its station's PLF ratio between its last full fiscal year (t-1) and
# ten years earlier (t-10). That equals the unit's own output ratio if the
# station's units run alike. It is clean only where the station's capacity
# was the same at t-10 and t-1 (no units added or removed in between, within
# 5%): the headline uses those stations; all others are shown as a check.
# Exit year: the fiscal year in which the station's listed capacity falls by
# at least half the unit's MW, from GEM's retirement year minus one onwards.
# Registered comparison: US median heat-input ratio 0.53 (820 unit exits).
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd

L = pd.read_csv(f'{DIR}/cea_units_long.csv', low_memory=False)
L['station'] = L['station'].astype(str).str.strip()
L['ver'] = L.groupby('book')['fy_end'].transform('max')
L['u0'] = L['unit'].astype(str).str.strip().isin(['0', '0.0'])
S = L[L['u0']].sort_values('ver')

# GEM plant -> CEA station names (checked by hand against CEA capacities)
MAP = {'Badarpur power station': ['BADARPUR'],
       'Barauni power station': ['BARAUNI'],
       'Bhusawal Thermal Power Station': ['BHUSAWAL'],
       'Bokaro Thermal Power Station': ['BOKARO B'],
       'Chandrapur Thermal Power Station': ['CHANDRAPUR', 'CHANDRAPUR_Coal'],
       'Chandrapura power station': ['CHANDRAPURA'],
       'Durgapur DVC power station': ['DURGAPUR'],
       'Durgapur Projects Limited power station': ['D.P.L.'],
       'Ennore Thermal Power Station': ['ENNORE'],
       'Gandhinagar Thermal Power Station': ['GANDHI NAGAR'],
       'GNDTP Bathinda power station': ['GNDTP(BHATINDA)'],
       'Kolaghat Thermal Power Station': ['KOLAGHAT'],
       'Koradi Thermal Power Station': ['KORADI'],
       'Korba Thermal power station': ['KORBA-EAST'],
       'Kothagudem Thermal Power Station': ['K_GUDEM'],
       'MP Amarkantak power station': ['AMAR KANTAK'],
       'Muzaffarpur power station': ['MUZAFFARPUR'],
       'Neyveli Thermal Power Station I': ['NEYVELI ST I'],
       'Panipat power station': ['PANIPAT'],
       'Panki Thermal Power Station': ['PANKI'],
       'Parichha power station': ['PARICHA'],
       'Parli Thermal Power Station': ['PARLI'],
       'Patratu JSEB power station': ['PATRATU'],
       'Ropar power station': ['ROPAR'],
       'Satpura Thermal Power Station': ['SATPURA'],
       'Sikka Thermal Power Station': ['SIKKA REP.'],
       'Talcher power station': ['TALCHER'],
       'Ukai Thermal Power Station': ['UKAI', 'UKAI_Coal']}
# Trombay unit 4 (150 MW) is listed by CEA as an oil/gas unit (TROMBAY_Oil):
# not a coal exit, excluded.
# Global Coal Plant Tracker (July 2026): coal units >= 100 MW, India, retired
# 2015-2025: plant, unit, MW, start year, retired year, state
GEM = [
    ('Badarpur power station', 'Phase 2 Unit 1', 210, 1978, 2018, 'Delhi'),
    ('Badarpur power station', 'Phase 2 Unit 2', 210, 1981, 2018, 'Delhi'),
    ('Barauni power station', 'Unit 6', 110, 1983, 2024, 'Bihar'),
    ('Barauni power station', 'Unit 7', 110, 1985, 2024, 'Bihar'),
    ('Bhusawal Thermal Power Station', 'Unit B2', 210, 1979, 2017, 'Maharashtra'),
    ('Bokaro Thermal Power Station', 'Bokaro B Unit 1', 210, 1986, 2017, 'Jharkhand'),
    ('Bokaro Thermal Power Station', 'Bokaro B Unit 2', 210, 1990, 2017, 'Jharkhand'),
    ('Bokaro Thermal Power Station', 'Bokaro B Unit 3', 210, 1993, 2021, 'Jharkhand'),
    ('Chandrapur Thermal Power Station', 'Unit 1', 210, 1983, 2016, 'Maharashtra'),
    ('Chandrapur Thermal Power Station', 'Unit 2', 210, 1984, 2016, 'Maharashtra'),
    ('Chandrapura power station', 'Unit 1', 130, 1964, 2017, 'Jharkhand'),
    ('Chandrapura power station', 'Unit 2', 130, 1964, 2017, 'Jharkhand'),
    ('Chandrapura power station', 'Unit 3', 130, 1968, 2020, 'Jharkhand'),
    ('Durgapur DVC power station', 'Unit 3', 140, 1966, 2016, 'West Bengal'),
    ('Durgapur DVC power station', 'Unit 4', 210, 1982, 2022, 'West Bengal'),
    ('Durgapur Projects Limited power station', 'Unit 6', 110, 1987, 2020, 'West Bengal'),
    ('Ennore Thermal Power Station', 'Unit 3', 110, 1972, 2017, 'Tamil Nadu'),
    ('Ennore Thermal Power Station', 'Unit 4', 110, 1973, 2017, 'Tamil Nadu'),
    ('Ennore Thermal Power Station', 'Unit 5', 110, 1975, 2017, 'Tamil Nadu'),
    ('Gandhinagar Thermal Power Station', 'Unit 1', 120, 1977, 2017, 'Gujarat'),
    ('Gandhinagar Thermal Power Station', 'Unit 2', 120, 1977, 2017, 'Gujarat'),
    ('GNDTP Bathinda power station', 'Unit 1', 110, 1974, 2018, 'Punjab'),
    ('GNDTP Bathinda power station', 'Unit 2', 110, 1975, 2018, 'Punjab'),
    ('GNDTP Bathinda power station', 'Unit 3', 120, 1978, 2018, 'Punjab'),
    ('GNDTP Bathinda power station', 'Unit 4', 120, 1979, 2018, 'Punjab'),
    ('Kolaghat Thermal Power Station', 'Unit 1', 210, 1990, 2022, 'West Bengal'),
    ('Kolaghat Thermal Power Station', 'Unit 2', 210, 1986, 2022, 'West Bengal'),
    ('Koradi Thermal Power Station', 'Unit 5', 200, 1978, 2017, 'Maharashtra'),
    ('Koradi Thermal Power Station', 'Unit 7', 210, 1983, 2021, 'Maharashtra'),
    ('Korba Thermal power station', 'Unit 5', 120, 1976, 2021, 'Chhattisgarh'),
    ('Korba Thermal power station', 'Unit 6', 120, 1981, 2021, 'Chhattisgarh'),
    ('Kothagudem Thermal Power Station', 'Unit 5', 120, 1974, 2020, 'Telangana'),
    ('Kothagudem Thermal Power Station', 'Unit 6', 120, 1974, 2019, 'Telangana'),
    ('Kothagudem Thermal Power Station', 'Unit 7', 120, 1977, 2020, 'Telangana'),
    ('Kothagudem Thermal Power Station', 'Unit 8', 120, 1978, 2019, 'Telangana'),
    ('MP Amarkantak power station', 'Phase II Unit 3', 120, 1977, 2015, 'Madhya Pradesh'),
    ('Muzaffarpur power station', 'Unit 1', 110, 1985, 2021, 'Bihar'),
    ('Muzaffarpur power station', 'Unit 2', 110, 1985, 2021, 'Bihar'),
    ('Neyveli Thermal Power Station I', 'Unit 7', 100, 1967, 2019, 'Tamil Nadu'),
    ('Neyveli Thermal Power Station I', 'Unit 8', 100, 1969, 2020, 'Tamil Nadu'),
    ('Neyveli Thermal Power Station I', 'Unit 9', 100, 1970, 2020, 'Tamil Nadu'),
    ('Panipat power station', 'Phase I Unit 1', 110, 1979, 2016, 'Haryana'),
    ('Panipat power station', 'Phase I Unit 2', 110, 1980, 2016, 'Haryana'),
    ('Panipat power station', 'Phase I Unit 3', 110, 1985, 2016, 'Haryana'),
    ('Panipat power station', 'Phase I Unit 4', 110, 1987, 2016, 'Haryana'),
    ('Panipat power station', 'Phase II Unit 5', 210, 1989, 2020, 'Haryana'),
    ('Panki Thermal Power Station', 'Unit 3', 105, 1976, 2018, 'Uttar Pradesh'),
    ('Panki Thermal Power Station', 'Unit 4', 105, 1977, 2018, 'Uttar Pradesh'),
    ('Parichha power station', 'Unit 1', 110, 1984, 2023, 'Uttar Pradesh'),
    ('Parichha power station', 'Unit 2', 110, 1985, 2023, 'Uttar Pradesh'),
    ('Parli Thermal Power Station', 'Unit 3', 210, 1982, 2016, 'Maharashtra'),
    ('Parli Thermal Power Station', 'Unit 4', 210, 1985, 2020, 'Maharashtra'),
    ('Parli Thermal Power Station', 'Unit 5', 210, 1987, 2020, 'Maharashtra'),
    ('Patratu JSEB power station', 'Unit 10', 110, 1986, 2017, 'Jharkhand'),
    ('Patratu JSEB power station', 'Unit 5', 100, 1971, 2017, 'Jharkhand'),
    ('Patratu JSEB power station', 'Unit 6', 100, 1972, 2017, 'Jharkhand'),
    ('Patratu JSEB power station', 'Unit 7', 110, 1977, 2017, 'Jharkhand'),
    ('Patratu JSEB power station', 'Unit 8', 110, 1978, 2017, 'Jharkhand'),
    ('Patratu JSEB power station', 'Unit 9', 110, 1984, 2017, 'Jharkhand'),
    ('Ropar power station', 'Unit 1', 210, 1984, 2018, 'Punjab'),
    ('Ropar power station', 'Unit 2', 210, 1985, 2018, 'Punjab'),
    ('Satpura Thermal Power Station', 'Unit 6', 200, 1979, 2025, 'Madhya Pradesh'),
    ('Satpura Thermal Power Station', 'Unit 7', 210, 1980, 2025, 'Madhya Pradesh'),
    ('Satpura Thermal Power Station', 'Unit 8', 210, 1983, 2025, 'Madhya Pradesh'),
    ('Satpura Thermal Power Station', 'Unit 9', 210, 1984, 2025, 'Madhya Pradesh'),
    ('Sikka Thermal Power Station', 'Unit 1', 120, 1988, 2017, 'Gujarat'),
    ('Sikka Thermal Power Station', 'Unit 2', 120, 1993, 2017, 'Gujarat'),
    ('Talcher power station', 'Unit B5', 110, 1982, 2021, 'Odisha'),
    ('Talcher power station', 'Unit B6', 110, 1983, 2021, 'Odisha'),
    ('Trombay power station', 'Unit 4', 150, 1965, 2016, 'Maharashtra'),
    ('Ukai Thermal Power Station', 'Unit 1', 120, 1976, 2017, 'Gujarat'),
    ('Ukai Thermal Power Station', 'Unit 2', 120, 1976, 2017, 'Gujarat'),
]
G = pd.DataFrame(GEM, columns=['plant', 'unit', 'mw', 'start', 'retired', 'state'])

def station_series(names):
    d = S[S['station'].isin(names)]
    gen = d.drop_duplicates('fy_end', keep='last').set_index('fy_end')['net_gen']
    cap = d.drop_duplicates('ver', keep='last').set_index('ver')['capacity']
    return gen.sort_index(), cap.sort_index()

def cap_end(cap, y):
    """Capacity at the end of fiscal year y (31 March y): the version ending y,
    else the nearest earlier version."""
    c = cap[cap.index <= y]
    return c.iloc[-1] if len(c) else np.nan

rows = []
for g in G.itertuples():
    if g.plant not in MAP:
        rows.append(dict(plant=g.plant, unit=g.unit, mw=g.mw, retired=g.retired,
                         note='not a CEA coal station (excluded)'))
        continue
    gen, cap = station_series(MAP[g.plant])
    # Headline dating: GEM retirement year R (calendar). The exit fiscal year
    # is taken as the one ending in March R+1, so t-1 is the fiscal year
    # ending March R and t-10 the one ending March R-9. Check dating: the
    # first fiscal year in which CEA's listed capacity falls by at least half
    # the unit's MW (CEA capacity lists lag closures and two versions are
    # missing, so this check dates some exits late).
    Yd = np.nan
    for y in range(int(g.retired) - 1, int(cap.index.max()) + 1):
        if cap_end(cap, y) <= cap_end(cap, y - 1) - 0.5 * g.mw:
            Yd = y; break
    out = dict(plant=g.plant, unit=g.unit, mw=g.mw, retired=g.retired,
               station=' / '.join(MAP[g.plant]), note='')
    for tag, Y in [('', int(g.retired) + 1), ('_capdrop', Yd)]:
        if Y != Y:
            out.update({f'ratio{tag}': np.nan}); continue
        t1, t10 = Y - 1, Y - 10
        c1, c10 = cap_end(cap, t1 - 1), cap_end(cap, t10 - 1)   # start of year
        window = [cap_end(cap, y) for y in range(t10 - 1, t1)]
        stable = bool(np.nanmax(window) <= 1.05 * np.nanmin(window))
        g1, g10 = gen.get(t1, np.nan), gen.get(t10, np.nan)
        p1 = g1 / (c1 * 8.76) if c1 and c1 > 0 else np.nan
        p10 = g10 / (c10 * 8.76) if c10 and c10 > 0 else np.nan
        out.update({f't1_fy_end{tag}': t1, f'cap_t10{tag}': c10,
                    f'cap_t1{tag}': c1, f'stable{tag}': stable,
                    f'plf_t10{tag}': p10, f'plf_t1{tag}': p1,
                    f'ratio{tag}': p1 / p10 if p10 and p10 > 0 else np.nan})
    rows.append(out)
R = pd.DataFrame(rows)
R.to_csv(f'{DIR}/outcomes/q5_india_units.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(R[['plant', 'unit', 'mw', 'retired', 'station', 't1_fy_end', 'cap_t10',
         'cap_t1', 'stable', 'plf_t10', 'plf_t1', 'ratio', 't1_fy_end_capdrop',
         'ratio_capdrop', 'note']].round(2).to_string(index=False))
def summ(d, lab, tag=''):
    d = d[d[f'ratio{tag}'].notna()]
    r, p10, p1 = d[f'ratio{tag}'], d[f'plf_t10{tag}'], d[f'plf_t1{tag}']
    st = d.drop_duplicates(['station', f't1_fy_end{tag}'])
    print(f'{lab}: {len(d)} units ({d.mw.sum():,.0f} MW) at {st.station.nunique()} '
          f'stations; median unit ratio {r.median():.2f} (IQR {r.quantile(.25):.2f}'
          f'-{r.quantile(.75):.2f}); MW-weighted mean {np.average(r, weights=d.mw):.2f}; '
          f'median PLF t-10 {100 * p10.median():.0f}%, t-1 {100 * p1.median():.0f}%; '
          f'units with ratio below 0.53: {(r < 0.53).mean():.0%}')
    print(f'    by station exit ({len(st)}): median {st[f"ratio{tag}"].median():.2f} '
          f'(IQR {st[f"ratio{tag}"].quantile(.25):.2f}-{st[f"ratio{tag}"].quantile(.75):.2f}); '
          f'median PLF t-10 {100 * st[f"plf_t10{tag}"].median():.0f}%, t-1 '
          f'{100 * st[f"plf_t1{tag}"].median():.0f}%')
print()
summ(R[R['stable'] == True], 'Headline: stations with unchanged capacity, GEM dating')
summ(R, 'All stations, GEM dating (check)')
summ(R[R['stable_capdrop'] == True], 'Stations with unchanged capacity, capacity-drop dating (check)', '_capdrop')
print('US benchmark: median unit heat-input ratio t-1/t-10 = 0.53 (820 exits)')
