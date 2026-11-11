# Q5, India, step 1 (post-registration; Q5 data had not been viewed at
# filing). Version 3 (10 Nov 2026). CEA's "CO2 Baseline Database for the
# Indian Power Sector" holds unit-wise net generation for five fiscal years
# per version. cea.nic.in cannot be reached from the UK or from Colab
# (version 2: 404 on every file id; the site does not open in a browser
# either), so step 1 takes the files from the Internet Archive's copies of
# the CEA pages: it lists every archived spreadsheet or zip under the CEA
# paths that have held the database and downloads the latest copy of each
# into MyDrive/coal_closures/cea_co2/. Files put in that folder by hand are
# read too. Step 2 prints an inventory and builds a long table of unit-year
# net generation (cea_units_long.csv) from header rows with fiscal years.
DIR = '/content/drive/MyDrive/coal_closures'
import os, re, io, time, zipfile, requests, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
OUT = f'{DIR}/cea_co2'
os.makedirs(OUT, exist_ok=True)
H = {'User-Agent': 'Mozilla/5.0 (research download; Internet Archive copies)'}

# ---------- 1. Internet Archive copies ----------
PREFIXES = ['cea.nic.in/reports/others/thermal/tpece/cdm_co2/',
            'cea.nic.in/old/reports/others/thermal/tpece/cdm_co2/',
            'cea.nic.in/wp-content/uploads/baseline/',
            'cea.nic.in/api/cdm-report-version/view-pdf/en/',
            'www.cea.nic.in/reports/others/thermal/tpece/cdm_co2/',
            'www.cea.nic.in/old/reports/others/thermal/tpece/cdm_co2/']
def cdx(prefix):
    for tries in range(4):
        try:
            r = requests.get('https://web.archive.org/cdx/search/cdx',
                             params=dict(url=prefix, matchType='prefix',
                                         output='json', limit=5000,
                                         filter='statuscode:200'),
                             headers=H, timeout=120)
            if r.status_code == 200:
                j = r.json()
                return pd.DataFrame(j[1:], columns=j[0]) if len(j) > 1 \
                    else pd.DataFrame()
        except Exception as e:
            print('   CDX retry:', type(e).__name__)
        time.sleep(5 * (tries + 1))
    return pd.DataFrame()
rows = []
for p in PREFIXES:
    d = cdx(p)
    print(f'  archive listing {p}: {len(d)} captures', flush=True)
    if len(d):
        rows.append(d)
C = pd.concat(rows) if rows else pd.DataFrame(columns=['original',
                                                         'timestamp'])
keep = C['original'].str.contains(r'\.(?:zip|xlsx?|xlsm)(?:\?|$)', case=False,
                                  regex=True) \
    | C['original'].str.contains('view-pdf/en/', regex=False)
keep &= ~C['original'].str.contains('user_guide|userguide', case=False)
C = C[keep].sort_values('timestamp').drop_duplicates('urlkey', keep='last')
print(f'  {len(C)} archived files to try')
got = 0
for r in C.itertuples():
    name = re.sub(r'[^A-Za-z0-9._-]', '_',
                  r.original.split('//')[-1].split('cea.nic.in/')[-1])[-120:]
    if any(f.startswith(name) for f in os.listdir(OUT)):
        got += 1; continue
    url = f'https://web.archive.org/web/{r.timestamp}id_/{r.original}'
    q = None
    for tries in range(4):                 # the Archive throttles bursts
        try:
            q = requests.get(url, headers=H, timeout=300)
            if q.status_code in (429, 500, 502, 503, 504):
                raise requests.ConnectionError(str(q.status_code))
            break
        except Exception as e:
            q = None
            time.sleep(20 * (tries + 1))
    if q is None:
        print('  fail after 4 tries', r.original); continue
    c = q.content
    if q.status_code != 200 or len(c) < 2000 or c[:4] == b'%PDF':
        continue
    if c[:2] == b'PK':
        ext = 'xlsx' if 'xl/workbook.xml' in zipfile.ZipFile(
            io.BytesIO(c)).namelist() else 'zip'
    elif c[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        ext = 'xls'
    else:
        continue
    open(f'{OUT}/{name}.{ext}', 'wb').write(c); got += 1
    print(f'  saved {name}.{ext} ({len(c) / 1e6:.1f} MB; archived '
          f'{r.timestamp[:8]})', flush=True)
    time.sleep(1)
print(f'Step 1: {got} database files in the folder.')

# ---------- 2. read every workbook in the folder ----------
for f in sorted(os.listdir(OUT)):
    if f.lower().endswith('.zip'):
        z = zipfile.ZipFile(f'{OUT}/{f}')
        for n in z.namelist():
            if re.search(r'\.(xlsx?|xlsm)$', n, re.I):
                q = f'{OUT}/{f[:-4]}__{re.sub(r"[^A-Za-z0-9._-]", "_", n.split("/")[-1])}'
                if not os.path.exists(q):
                    open(q, 'wb').write(z.read(n))
books = sorted(f for f in os.listdir(OUT)
               if re.search(r'\.(xlsx?|xlsm)$', f, re.I))
print(f'\nStep 2: {len(books)} workbooks')
FY = re.compile(r'(19|20)(\d\d)\s*[-/]\s*(\d{2,4})')
rows = []
for b in books:
    try:
        x = pd.ExcelFile(f'{OUT}/{b}')
    except Exception as e:
        print(f'  {b}: cannot open ({str(e)[:60]})'); continue
    ver = re.search(r'ver\w*?[_ ]?(\d+)', b, re.I)
    print(f'\n  {b}: sheets {x.sheet_names}')
    sheets = [n for n in x.sheet_names if n.strip().lower() == 'data'] or \
        [n for n in x.sheet_names if 'data' in n.lower() or 'unit' in n.lower()]
    for s in sheets:
        d = x.parse(s, header=None)
        if d.shape[0] < 20:
            continue
        best = None
        for h in range(min(25, len(d))):
            cells = d.iloc[h].astype(str)
            nfy = cells.str.contains(FY).sum()
            if nfy >= 3 and (best is None or nfy > best[1]):
                best = (h, nfy)
        if best is None:
            continue
        h = best[0]
        hdr = [' '.join(str(d.iat[k, j]) for k in range(max(0, h - 2), h + 1)
                        if str(d.iat[k, j]) != 'nan') for j in range(d.shape[1])]
        print(f'    sheet {s!r}: {d.shape}, header row {h}:',
              [c[:30] for c in hdr[:24]])
        low = [c.lower() for c in hdr]
        c_name = next((j for j, c in enumerate(low) if 'name' in c or
                       'station' in c), None)
        c_unit = next((j for j, c in enumerate(low) if re.search(r'unit[_ ]?no', c)),
                      None)
        c_comm = next((j for j, c in enumerate(low) if 'comm' in c), None)
        c_state = next((j for j, c in enumerate(low) if c.strip() == 'state'), None)
        c_cap = next((j for j, c in enumerate(low) if 'capacity' in c
                      or 'mw' in c), None)
        c_fuel = next((j for j, c in enumerate(low) if 'fuel' in c), None)
        c_type = next((j for j, c in enumerate(low) if c.strip() == 'type'
                       or 'type' in c), None)
        gen = [(j, FY.search(hdr[j])) for j in range(len(hdr))
               if FY.search(hdr[j]) and re.search(r'gen', low[j])]
        if c_name is None or not gen:
            continue
        body = d.iloc[h + 1:]
        for j, m in gen:
            y_end = int(m.group(1) + m.group(2)) + 1
            v = pd.to_numeric(body.iloc[:, j], errors='coerce')
            rows.append(pd.DataFrame({
                'book': b, 'sheet': s, 'station': body.iloc[:, c_name].astype(str),
                'unit': body.iloc[:, c_unit].astype(str) if c_unit is not None else '',
                'capacity': pd.to_numeric(body.iloc[:, c_cap], errors='coerce')
                if c_cap is not None else np.nan,
                'fuel': body.iloc[:, c_fuel].astype(str) if c_fuel is not None else '',
                'comm': body.iloc[:, c_comm].astype(str) if c_comm is not None else '',
                'state': body.iloc[:, c_state].astype(str) if c_state is not None else '',
                'fy_end': y_end, 'net_gen': v}).dropna(subset=['net_gen']))
L = pd.concat(rows) if rows else pd.DataFrame()
L.to_csv(f'{DIR}/cea_units_long.csv', index=False)
print(f'\nLong table: {len(L)} unit-years from {L["book"].nunique() if len(L) else 0}'
      f' workbooks; fiscal years ending '
      f'{L.fy_end.min() if len(L) else "-"} to {L.fy_end.max() if len(L) else "-"}.')
if len(L):
    print(L.groupby('book')['fy_end'].agg(['min', 'max', 'size']).to_string())
    print(L.sample(min(8, len(L)), random_state=1).to_string())
print('\nDone. Send this output.')
