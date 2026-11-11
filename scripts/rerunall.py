# Rerun the whole US chain on the complete EPA annual file.
# 1) swap files: old incomplete file -> us_epa_annual_all_OLD.csv,
#    complete year-by-year file -> us_epa_annual_all.csv
# 2) run, in order: usunits, usevents, isrmrows, valuepred, val2010,
#    step1pm, prefiling. No air-quality values near exit events are read.
DIR = '/content/drive/MyDrive/coal_closures'
import os, time

cur, old, v2 = (f'{DIR}/us_epa_annual_all.csv',
                f'{DIR}/us_epa_annual_all_OLD.csv',
                f'{DIR}/us_epa_annual_all_v2.csv')
if os.path.exists(v2):
    if not os.path.exists(old):
        os.rename(cur, old)
    elif os.path.exists(cur):
        os.remove(cur)
    os.rename(v2, cur)
    print('Swapped in the complete annual file')
else:
    print('Complete file already in place')

for s in ['usunits.py', 'usevents.py', 'isrmrows.py', 'valuepred.py',
          'val2010.py', 'step1pm.py', 'prefiling.py']:
    path = f'{DIR}/{s}'
    print('\n' + '=' * 70 + f'\nRUNNING {s}\n' + '=' * 70, flush=True)
    t0 = time.time()
    code = compile(open(path).read(), path, 'exec')
    exec(code, {'__name__': '__main__'})
    print(f'[{s} done in {(time.time() - t0) / 60:.1f} min]', flush=True)
