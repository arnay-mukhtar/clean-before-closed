# Step: download the InMAP Source-Receptor Matrix and print its structure.
# Reads no air-quality outcomes. Saves to the Colab disk, not Drive.
import os, shutil, zipfile, time, requests

WORK = '/content/isrm'
os.makedirs(WORK, exist_ok=True)
URL = ('https://zenodo.org/records/3590127/files/'
       'isrm_v1.2.1.zip?download=1')
ZIP = f'{WORK}/isrm_v1.2.1.zip'

free = shutil.disk_usage('/content').free / 1e9
print(f'Free disk: {free:.0f} GB')
if free < 35:
    raise SystemExit('Need about 35 GB free. Runtime > Disconnect and '
                     'delete runtime, then run again.')

if not os.path.exists(ZIP):
    t0 = time.time()
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0)) / 1e9
        print(f'Downloading {total:.1f} GB')
        got, nxt = 0, 1e9
        with open(ZIP + '.part', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                got += len(chunk)
                if got >= nxt:
                    mins = (time.time() - t0) / 60
                    print(f'  {got / 1e9:.0f} GB, {mins:.1f} min')
                    nxt += 1e9
    os.rename(ZIP + '.part', ZIP)
    print('Download done')

with zipfile.ZipFile(ZIP) as z:
    print('\nFiles in the archive:')
    for i in z.infolist():
        print(f'  {i.filename}  {i.file_size / 1e9:.2f} GB '
              f'(zipped {i.compress_size / 1e9:.2f} GB)')
    need = sum(i.file_size for i in z.infolist()) / 1e9
    free = shutil.disk_usage('/content').free / 1e9
    print(f'Unzipped size {need:.1f} GB, free {free:.0f} GB')
    if need < free - 2:
        z.extractall(WORK)
        print('Extracted')
    else:
        raise SystemExit('Not enough space to extract. Send me this output.')

import netCDF4
for root, _, files in os.walk(WORK):
    for fn in files:
        if fn.endswith(('.ncf', '.nc', '.nc4')):
            p = os.path.join(root, fn)
            print('\n=====', p)
            ds = netCDF4.Dataset(p)
            print('Dimensions:')
            for k, d in ds.dimensions.items():
                print(f'  {k}: {len(d)}')
            print('Variables:')
            for k, v in ds.variables.items():
                ch = v.chunking() if hasattr(v, 'chunking') else None
                print(f'  {k} {v.dimensions} {v.dtype} chunks={ch}')
                for a in v.ncattrs():
                    print(f'      {a}: {str(v.getncattr(a))[:120]}')
            print('Global attributes:')
            for a in ds.ncattrs():
                print(f'  {a}: {str(ds.getncattr(a))[:200]}')
            ds.close()
        elif not fn.endswith('.zip'):
            print('\nOther file:', os.path.join(root, fn))
