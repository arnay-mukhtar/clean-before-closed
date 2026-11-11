# Step: get EEA plant-level data for Europe's large combustion plants
# (annual SO2, NOx, dust and fuel input, 2016-2023) and show its layout.
# Operation data only; no air-quality outcomes.
DIR = '/content/drive/MyDrive/coal_closures'
import os, io, re, zipfile, requests
import pandas as pd
from urllib.parse import unquote, urljoin

TOKEN = 'sptXqwkQr5g7Bp5'
FOLDER = 'eea_t_ied-eprtr_p_2007-2023_v14_r00'
OUT = f'{DIR}/eea'
os.makedirs(OUT, exist_ok=True)
AUTH = (TOKEN, '')
ROOTS = ['https://sdi.eea.europa.eu/datashare/public.php/webdav/',
         f'https://sdi.eea.europa.eu/datashare/public.php/dav/files/{TOKEN}/']

def listdir(url):
    r = requests.request('PROPFIND', url, headers={'Depth': '1'},
                         auth=AUTH, timeout=60)
    r.raise_for_status()
    hrefs = re.findall(r'<(?:\w+:)?href>([^<]+)</(?:\w+:)?href>',
                       r.text, re.I)
    out = []
    for h in hrefs:
        full = urljoin(url, h)
        if full.rstrip('/') != url.rstrip('/'):
            out.append(full)
    return out

found, seen = [], set()
def walk(url, depth=0):
    if depth > 3 or url in seen:
        return
    seen.add(url)
    for u in listdir(url):
        name = unquote(u.rstrip('/').split('/')[-1])
        print('  ' * depth + name)
        if u.endswith('/'):
            walk(u, depth + 1)
        elif re.search(r'F5_[12]', name):
            found.append(u)

ok = False
for root in ROOTS:
    try:
        print('Trying', root)
        walk(root + FOLDER + '/')
        ok = True
        break
    except Exception as e:
        print('  failed:', str(e)[:150])
        seen.clear(); found.clear()

if not ok or not found:
    # fallback: the dataset's direct download (zip of all tables)
    z = f'{OUT}/eea_ied_v14_all.zip'
    if not os.path.exists(z):
        print('Falling back to the full dataset download')
        r = requests.get('https://sdi.eea.europa.eu/data/'
                         '21e758c6-a9ac-4a7d-a64a-19d2ba9eecb7',
                         timeout=1800, stream=True)
        r.raise_for_status()
        print('Content type:', r.headers.get('content-type'),
              ' size:', r.headers.get('content-length'))
        with open(z, 'wb') as f:
            for ch in r.iter_content(8 << 20):
                f.write(ch)
    if not zipfile.is_zipfile(z):
        print(open(z, 'rb').read(800))
        raise SystemExit('Download is not a zip. Send me this output.')
    zz = zipfile.ZipFile(z)
    print('Files in zip:')
    for n in zz.namelist():
        print(' ', n)
        if re.search(r'F5_[12]', n) and not n.endswith('/'):
            open(f'{OUT}/' + os.path.basename(n), 'wb').write(zz.read(n))

for u in found:
    name = unquote(u.split('/')[-1])
    path = f'{OUT}/{name}'
    if not os.path.exists(path):
        r = requests.get(u, auth=AUTH, timeout=600)
        r.raise_for_status()
        open(path, 'wb').write(r.content)
    print(f'\nSaved {name}: {os.path.getsize(path) / 1e6:.1f} MB')

def read_any(path):
    if path.endswith('.zip'):
        z = zipfile.ZipFile(path)
        inner = [n for n in z.namelist() if n.endswith(('.csv', '.xlsx'))]
        return {n: read_any_bytes(z.read(n), n) for n in inner}
    return {os.path.basename(path): read_any_bytes(open(path, 'rb').read(),
                                                   path)}

def read_any_bytes(b, name):
    if name.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(b))
    for enc in ['utf-8', 'utf-16', 'latin-1']:
        for sep in [',', ';', '\t']:
            try:
                d = pd.read_csv(io.BytesIO(b), sep=sep, encoding=enc,
                                low_memory=False)
                if d.shape[1] > 3:
                    return d
            except Exception:
                pass
    raise ValueError('could not read ' + name)

pd.set_option('display.width', 250, 'display.max_columns', 60)
for fn in sorted(os.listdir(OUT)):
    if not re.search(r'F5_2', fn):
        continue
    for n, d in read_any(f'{OUT}/{fn}').items():
        print(f'\n===== {n}: {d.shape}')
        print('Columns:', list(d.columns))
        print(d.head(3).T.to_string())
        for c in d.columns:
            if d[c].dtype == object and 1 < d[c].nunique() <= 40:
                print(f'\n{c}:', d[c].value_counts().head(40).to_dict())
