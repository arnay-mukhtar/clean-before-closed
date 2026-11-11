# Figure 9 (secondary NO2 test, Section 6.9). Reads
# outcomes/no2_secondary_events.csv and no2_secondary_placebo.csv (batchE.py).
# Panel A: each exit's control-adjusted fall in the TROPOMI NO2 contrast
# (log near 0-10 km minus log ring 25-50 km, year before minus year after)
# against its NOx drop, US and Europe, with the through-origin slope.
# Panel B: the exits' falls beside the placebo changes at never-exit plants
# (each centred on its region-year mean), with the mean and its 95%
# bootstrap interval. Saves outcomes/fig9_no2.png (300 dpi) and .pdf.
import os, sys, hashlib
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') \
    else '/content/drive/MyDrive/coal_closures'
OUT = f'{DIR}/outcomes'
ev = pd.read_csv(f'{OUT}/no2_secondary_events.csv')
pl = pd.read_csv(f'{OUT}/no2_secondary_placebo.csv')
pl['c'] = pl['dC'] - pl.groupby(['region', 'year'])['dC'].transform('mean')
ev['kt'] = ev['nox_drop'] / 1e3
b = (ev['kt'] * ev['fall']).sum() / (ev['kt'] ** 2).sum()
# the interval is read from batchE.py's output, so the figure and the text
# report the same bootstrap
S_ = pd.read_csv(f'{OUT}/batchE_summary.csv').set_index('analysis')
r_ = S_.loc['NO2 contrast, US and Europe (log points)']
m, lo, hi = ev['fall'].mean(), r_['lo_boot'], r_['hi_boot']
print(f'{len(ev)} exits (US {int((ev.region == "US").sum())}, Europe '
      f'{int((ev.region == "EU").sum())}); mean fall {m:.3f} [{lo:.3f}, '
      f'{hi:.3f}]; slope {b:.4f} per kt; {len(pl)} placebo changes')

INK, MUTED, GRID = '#222222', '#6b6b66', '#e6e6e1'
US_C, EU_C, PL_C = '#2a78d6', '#eb6834', '#b9b9b2'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                               gridspec_kw={'width_ratios': [1.25, 1]})
# ---- panel A ----
ax1.axhline(0, color=MUTED, lw=0.8)
xs = np.linspace(0, ev['kt'].max() * 1.05, 50)
ax1.plot(xs, b * xs, color=INK, lw=1.5, zorder=2)
for reg, col, mk, lab in [('US', US_C, 'o', 'US exits'),
                          ('EU', EU_C, 's', 'European exits')]:
    d = ev[ev.region == reg]
    ax1.scatter(d['kt'], d['fall'], s=34, c=col, marker=mk, edgecolors='white',
                linewidths=1.0, zorder=3, label=f'{lab} ({len(d)})')
ax1.text(xs[-1], b * xs[-1], f'  slope {b:.3f}\n  per 1,000 t', color=INK,
         va='center', ha='left', fontsize=8)
ax1.set_xlabel('NOx drop at the exit, thousand tons (year before to year after)')
ax1.set_ylabel('Fall in near-field NO2 contrast (log points)')
ax1.set_xlim(0, xs[-1] * 1.22)
ax1.grid(axis='y', color=GRID, lw=0.6); ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc='upper left', fontsize=8)
ax1.set_title('A. Fall at each exit against its NOx cut', loc='left',
              fontsize=10, color=INK)
# ---- panel B ----
j = np.random.default_rng(1)
ax2.axvline(0, color=MUTED, lw=0.8)
ax2.scatter(pl['c'], 0 + j.uniform(-0.22, 0.22, len(pl)), s=6, c=PL_C,
            linewidths=0, zorder=2)
for reg, col, mk in [('US', US_C, 'o'), ('EU', EU_C, 's')]:
    d = ev[ev.region == reg]
    ax2.scatter(d['fall'], 1 + j.uniform(-0.22, 0.22, len(d)), s=22, c=col,
                marker=mk, edgecolors='white', linewidths=0.8, zorder=3)
ax2.errorbar(m, 1.42, xerr=[[m - lo], [hi - m]], fmt='D', color=INK,
             ms=6, capsize=3, lw=1.5, zorder=4)
ax2.text(hi, 1.42, f'  mean {m:.3f}\n  [{lo:.3f}, {hi:.3f}]', va='center',
         fontsize=8, color=INK)
ax2.set_yticks([0, 1])
ax2.set_yticklabels([f'Never-exit plants\n({len(pl)} placebo changes,\ncentred)',
                     f'Exits ({len(ev)})'])
ax2.set_ylim(-0.5, 1.75)
ax2.set_xlabel('Fall in near-field NO2 contrast (log points)')
ax2.grid(axis='x', color=GRID, lw=0.6); ax2.set_axisbelow(True)
ax2.set_title('B. Exits against placebo changes', loc='left', fontsize=10,
              color=INK)
fig.suptitle('Figure 9. Satellite NO2 near coal exits from 2019, US and Europe '
             '(secondary registered test)', x=0.01, ha='left', fontsize=11,
             color=INK)
fig.text(0.01, -0.02, 'NO2 contrast: log mean TROPOMI tropospheric column '
         'within 10 km of the plant minus the 25-50 km ring. Exits are '
         'control-adjusted by the mean placebo change for their region and '
         'year. Interval: bootstrap over exits. Sources: Copernicus '
         'Sentinel-5P (ESA), EPA CAMPD, EEA Industrial Reporting.',
         fontsize=7.5, color=MUTED, wrap=True)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
for ext in ['png', 'pdf']:
    p = f'{OUT}/fig9_no2.{ext}'
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(p, hashlib.sha256(open(p, 'rb').read()).hexdigest())
if 'ipykernel' in sys.modules:
    plt.show()
