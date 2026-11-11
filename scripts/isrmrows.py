# Step: (1) InMAP damage-per-tonne file; (2) InMAP matrix rows for the grid
# cell of every US plant that ever burned coal (about 500), saved to Drive.
# If the 66 GB extract from earlier is gone, this re-downloads the zip
# (about 32 min) and reads only the needed rows in one pass (about 20 min).
DIR = '/content/drive/MyDrive/coal_closures'
WORK = '/content/isrm'
import os, io, json, time, zipfile, shutil, requests
import numpy as np, pandas as pd
from pyproj import Transformer

# ---------- (1) marginal damages ----------
p_mv = f'{DIR}/isrm_marginal_values.csv'
if not os.path.exists(p_mv):
    meta = requests.get('https://zenodo.org/api/records/3590127',
                        timeout=60).json()
    f = [f for f in meta['files'] if f['key'].startswith('marginal')][0]
    link = f['links'].get('self') or f['links'].get('download')
    z = zipfile.ZipFile(io.BytesIO(requests.get(link, timeout=600).content))
    print('Inside marginal zip:', z.namelist())
    inner = [n for n in z.namelist() if n.endswith('.csv')][0]
    open(p_mv, 'wb').write(z.read(inner))
mv = pd.read_csv(p_mv)
print('Marginal values:', mv.shape)
print(list(mv.columns))
print(mv.head(3).T.to_string())

# ---------- (2) which source cells we need ----------
G = np.load(f'{DIR}/isrm_grid.npz')
W, E, S, N = G['W'], G['E'], G['S'], G['N']
NS = len(W)
LCC = ('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 '
       '+y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
tr = Transformer.from_crs('EPSG:4326', LCC, always_xy=True)
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'primaryfuelinfo'])
coal_fids = set(a.loc[a['primaryfuelinfo'].fillna('').str.contains('Coal'),
                      'facilityid'].astype(int))
fac = pd.read_csv(f'{DIR}/us_epa_facilities.csv')
fac = fac[fac['facilityid'].isin(coal_fids)].copy()
x, y = tr.transform(fac['longitude'].values, fac['latitude'].values)
cell = []
for xi, yi in zip(x, y):
    h = np.where((W <= xi) & (xi < E) & (S <= yi) & (yi < N))[0]
    cell.append(int(h[0]) if len(h) else -1)
fac['cell'] = cell
fac[['facilityid', 'facilityname', 'statecode', 'latitude', 'longitude',
     'cell']].to_csv(f'{DIR}/isrm_coal_facility_cells.csv', index=False)
cells = np.array(sorted(set(fac['cell']) - {-1}))
print(f'\nCoal facilities: {len(fac)}, outside grid: {(fac.cell < 0).sum()}, '
      f'distinct source cells: {len(cells)}')

SPECIES = {'PrimaryPM25': 32962963760, 'pSO4': 98888873864,
           'pNO3': 131851828916}           # byte offsets from the header
LAYERS = [1, 2]
ROW = NS * 4
out = {f'{sp}_L{L}': np.zeros((len(cells), NS), '>f4')
       for sp in SPECIES for L in LAYERS}

if os.path.exists(f'{WORK}/out/meta.json'):
    print('Using the earlier extract on this runtime')
    for sp in SPECIES:
        for L in LAYERS:
            m = np.memmap(f'{WORK}/out/{sp}_L{L}.bin', dtype='>f4', mode='r',
                          shape=(NS, NS))
            out[f'{sp}_L{L}'][:] = m[cells]
else:
    ZIP = f'{WORK}/isrm_v1.2.1.zip'
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(ZIP):
        print('Re-downloading the matrix zip (12.5 GB)')
        t0 = time.time()
        URL = ('https://zenodo.org/records/3590127/files/'
               'isrm_v1.2.1.zip?download=1')
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            got, nxt = 0, 2e9
            with open(ZIP + '.part', 'wb') as f:
                for ch in r.iter_content(8 << 20):
                    f.write(ch); got += len(ch)
                    if got >= nxt:
                        print(f'  {got / 1e9:.0f} GB, '
                              f'{(time.time() - t0) / 60:.0f} min', flush=True)
                        nxt += 2e9
        os.rename(ZIP + '.part', ZIP)
    jobs = []
    for sp, base in SPECIES.items():
        for L in LAYERS:
            for i, c in enumerate(cells):
                jobs.append((base + (L * NS + int(c)) * ROW,
                             f'{sp}_L{L}', i))
    jobs.sort()
    zf = zipfile.ZipFile(ZIP)
    name = [n for n in zf.namelist() if n.endswith('.ncf')][0]
    t0, pos, nxt = time.time(), 0, 10e9
    with zf.open(name) as f:
        for start, key, i in jobs:
            while pos < start:
                n = len(f.read(min(64 << 20, start - pos)))
                if n == 0:
                    raise SystemExit('Unexpected end of file')
                pos += n
                if pos >= nxt:
                    print(f'  {pos / 1e9:.0f} of 165 GB, '
                          f'{(time.time() - t0) / 60:.1f} min', flush=True)
                    nxt += 10e9
            buf = f.read(ROW)
            pos += len(buf)
            out[key][i] = np.frombuffer(buf, '>f4')

np.savez_compressed(f'{DIR}/isrm_rows_allcoal.npz', cells=cells,
                    **{k: v.astype('<f4') for k, v in out.items()})
print(f'\nSaved isrm_rows_allcoal.npz '
      f'({os.path.getsize(f"{DIR}/isrm_rows_allcoal.npz") / 1e6:.0f} MB)')

# sanity check against the earlier 217-cell file
old = np.load(f'{DIR}/isrm_rows_plants.npz')
common = np.intersect1d(old['cells'], cells)
i_old = np.searchsorted(old['cells'], common)
i_new = np.searchsorted(cells, common)
diff = np.abs(old['pSO4_L2'][i_old].astype('f8')
              - out['pSO4_L2'][i_new].astype('f8')).max()
print(f'Check vs earlier rows on {len(common)} shared cells: '
      f'max difference {diff:.2e} (should be 0)')
