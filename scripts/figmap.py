# Figure 4 (design map). Reads outcomes/map_points.csv (batchB.py) and the
# Census Bureau's 1:20m state boundaries (cb_2018_us_state_20m), and draws
# the contiguous US in Albers equal-area projection: all 502 US coal plant
# exit events (grey, sized by SO2 drop), the 51 events in the primary PM2.5
# sample (blue), PM2.5 monitors within 35 km of those events (dark) and
# background monitors more than 50 km from any coal plant (light).
# Saves outcomes/fig4_map.png (300 dpi) and fig4_map.pdf.
DIR = '/content/drive/MyDrive/coal_closures'
import os, requests, hashlib
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt

P = pd.read_csv(f'{DIR}/outcomes/map_points.csv')
p_st = f'{DIR}/cb_2018_us_state_20m.zip'
if not os.path.exists(p_st):
    r = requests.get('https://www2.census.gov/geo/tiger/GENZ2018/shp/'
                     'cb_2018_us_state_20m.zip', timeout=120)
    r.raise_for_status(); open(p_st, 'wb').write(r.content)
ST = gpd.read_file(f'zip://{p_st}')
ST = ST[~ST['STUSPS'].isin(['AK', 'HI', 'PR'])].to_crs(5070)
def pts(d):
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d['lon'], d['lat']),
                            crs=4326).to_crs(5070)
CONUS = lambda d: d[d['lat'].between(24, 50) & d['lon'].between(-125, -66)]
ev = pts(CONUS(P[P['kind'] == 'exit event']).copy())   # all 502 are CONUS
mo = pts(CONUS(P[P['kind'] == 'PM2.5 monitor']).copy())  # drops AK, HI, PR
print(f"{len(ev)} exit events, {int(ev['pm25_sample'].astype(str).str.lower().eq('true').sum())} in the PM2.5 test; {len(mo)} monitors in the contiguous US")
BLUE, GREY, DARK = '#2a6fd6', '#9a9a92', '#333333'
fig, ax = plt.subplots(figsize=(10, 6.2))
ST.plot(ax=ax, color='white', edgecolor='#c9c9c4', linewidth=0.5)
bg = mo[mo['role'].str.startswith('background')]
nr = mo[mo['role'].str.startswith('near')]
ax.scatter(bg.geometry.x, bg.geometry.y, s=3, c='#bdbdb6', marker='o',
           linewidths=0, label=f'Background PM2.5 monitors ({len(bg)})')
SMAX = max(ev['so2_drop'].max(), 1)      # one scale for every marker
size = lambda s: 6 + 60 * np.sqrt(np.clip(s.fillna(0), 0, None) / SMAX)
insmp = ev['pm25_sample'].astype(str).str.lower().eq('true')
oth = ev[~insmp]
smp = ev[insmp]
ax.scatter(oth.geometry.x, oth.geometry.y, s=size(oth['so2_drop']), c=GREY,
           alpha=0.55, linewidths=0, label=f'Plant exit events ({len(ev)})')
ax.scatter(smp.geometry.x, smp.geometry.y, s=size(smp['so2_drop']), c=BLUE,
           alpha=0.9, linewidths=0.4, edgecolors='white',
           label=f'Events in the PM2.5 test ({len(smp)})')
ax.scatter(nr.geometry.x, nr.geometry.y, s=7, c=DARK, marker='^',
           linewidths=0, label=f'PM2.5 monitors within 35 km ({len(nr)})')
xmin, ymin, xmax, ymax = ST.total_bounds
ax.set_xlim(xmin - 5e4, xmax + 5e4); ax.set_ylim(ymin - 5e4, ymax + 5e4)
ax.set_axis_off()
ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.02), frameon=False,
          fontsize=8.5, markerscale=1.2)
ax.set_title('Figure 4. US coal plant exit events, 2005–2024, and the '
             'monitors used in the pre-registered test', fontsize=11,
             loc='left')
fig.text(0.01, 0.01, 'Marker area proportional to the event\'s SO2 drop. '
         'Sources: EPA CAMPD, EPA AQS, US Census Bureau (cb_2018_us_state_20m).',
         fontsize=7.5, color='#666666')
fig.tight_layout()
for ext in ['png', 'pdf']:
    p = f'{DIR}/outcomes/fig4_map.{ext}'
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(p, hashlib.sha256(open(p, 'rb').read()).hexdigest())
plt.show()
