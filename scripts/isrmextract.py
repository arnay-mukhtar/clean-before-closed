# Step: pull the parts of the InMAP matrix we need straight out of the zip,
# without unzipping the full 165 GB file. Reads no air-quality outcomes.
# Keeps: sulfate, nitrate and primary PM2.5 responses (the species coal
# plants emit: SO2, NOx, primary PM) for the low- and high-stack layers,
# plus every small variable (grid cell edges, population and so on).
import os, io, json, time, struct, shutil, zipfile
import numpy as np

WORK = '/content/isrm'
ZIP = f'{WORK}/isrm_v1.2.1.zip'
OUT = f'{WORK}/out'
os.makedirs(OUT, exist_ok=True)
TARGETS = ['psO4', 'pNO3', 'PrimaryPM25']
TARGETS = [t.lower() for t in TARGETS]
KEEP_LAYERS = [1, 2]          # 0 ground, 1 low stack, 2 high stack
SMALL = 1.5e9                 # variables below this size are kept whole
CHUNK = 64 << 20

if os.path.exists(f'{OUT}/meta.json'):
    raise SystemExit('Already extracted. Files are in ' + OUT)

zf = zipfile.ZipFile(ZIP)
name = [n for n in zf.namelist() if n.endswith('.ncf')][0]
total = zf.getinfo(name).file_size

# ---------- read and parse the netCDF classic header ----------
with zf.open(name) as f:
    head = f.read(32 << 20)
if head[:4] == b'\x89HDF':
    raise SystemExit('File is netCDF-4/HDF5, not classic. Send me this.')
if head[:3] != b'CDF':
    raise SystemExit(f'Unknown format, first bytes {head[:8]!r}')
ver = head[3]
print('netCDF classic version', ver)
B = io.BytesIO(head)
B.read(4)
cnt = '>q' if ver == 5 else '>i'
csz = 8 if ver == 5 else 4

def rd(fmt):
    return struct.unpack(fmt, B.read(struct.calcsize(fmt)))[0]

def rcount():
    return rd(cnt)

def pad4(n):
    return (4 - n % 4) % 4

def rname():
    n = rcount()
    s = B.read(n).decode('utf-8', 'replace')
    B.read(pad4(n))
    return s

TSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}
TCODE = {1: 'i1', 2: 'S1', 3: '>i2', 4: '>i4', 5: '>f4', 6: '>f8',
         7: 'u1', 8: '>u2', 9: '>u4', 10: '>i8', 11: '>u8'}

def ratts():
    tag = rd('>i')
    n = rcount()
    out = {}
    for _ in range(n):
        an = rname()
        t = rd('>i')
        k = rcount()
        nb = k * TSIZE[t]
        raw = B.read(nb)
        B.read(pad4(nb))
        if t == 2:
            out[an] = raw.decode('utf-8', 'replace')
        else:
            v = np.frombuffer(raw, TCODE[t])
            out[an] = v.tolist() if v.size > 1 else v.item()
    return out

numrecs = rcount()
tag = rd('>i')
ndim = rcount()
dims = []
for _ in range(ndim):
    dn = rname()
    dl = rcount()
    dims.append((dn, dl))
gatts = ratts()
tag = rd('>i')
nvar = rcount()
vars_ = []
for _ in range(nvar):
    vn = rname()
    nd = rcount()
    ids = [rcount() for _ in range(nd)]
    va = ratts()
    t = rd('>i')
    vsize = rcount()
    begin = rd('>q') if ver in (2, 5) else rd('>i')
    vars_.append(dict(name=vn, dimids=ids, atts=va, type=t,
                      vsize=vsize, begin=begin))

print('Records:', numrecs)
print('Dimensions:', dims)
print('Global attributes:')
for k, v in gatts.items():
    print(f'  {k}: {str(v)[:300]}')

rec_dim = [i for i, (_, l) in enumerate(dims) if l == 0]
rec_dim = rec_dim[0] if rec_dim else None
recsize = 0
for v in vars_:
    shp = [numrecs if i == rec_dim else dims[i][1] for i in v['dimids']]
    v['shape'] = shp
    v['dimnames'] = [dims[i][0] for i in v['dimids']]
    v['is_rec'] = bool(v['dimids']) and v['dimids'][0] == rec_dim
    per = TSIZE[v['type']] * int(np.prod(shp[1:] if v['is_rec'] else shp))
    v['slab'] = per
    if v['is_rec']:
        recsize += per + pad4(per)
print('\nVariables:')
for v in vars_:
    gb = v['slab'] * (numrecs if v['is_rec'] else 1) / 1e9
    print(f"  {v['name']} {v['dimnames']} {v['shape']} {TCODE[v['type']]}"
          f" {gb:.2f} GB begin={v['begin']} rec={v['is_rec']}")
    for k, a in v['atts'].items():
        print(f'      {k}: {str(a)[:150]}')

# ---------- plan which byte ranges to keep ----------
jobs = []   # (start, nbytes, filename)
meta = dict(dims=dims, gatts=gatts, layers_kept=KEEP_LAYERS, vars={})
for v in vars_:
    tc = TCODE[v['type']]
    full = v['slab'] * (numrecs if v['is_rec'] else 1)
    if v['is_rec']:
        slabs = [(v['begin'] + r * recsize, v['slab'], r)
                 for r in range(numrecs)]
        sub = v['shape'][1:]
    elif full > SMALL and v['shape'][0] <= 10:
        step = full // v['shape'][0]
        slabs = [(v['begin'] + k * step, step, k)
                 for k in range(v['shape'][0])]
        sub = v['shape'][1:]
    else:
        slabs = [(v['begin'], full, None)]
        sub = v['shape']
    if full <= SMALL:
        fn = f"{OUT}/{v['name']}.bin"
        for s, n, k in slabs:
            jobs.append((s, n, fn))
        meta['vars'][v['name']] = dict(file=fn, dtype=tc, shape=v['shape'],
                                       dimnames=v['dimnames'],
                                       atts=v['atts'])
    elif v['name'].lower() in TARGETS:
        for s, n, k in slabs:
            if k in KEEP_LAYERS:
                fn = f"{OUT}/{v['name']}_L{k}.bin"
                jobs.append((s, n, fn))
                meta['vars'][f"{v['name']}_L{k}"] = dict(
                    file=fn, dtype=tc, shape=sub,
                    dimnames=v['dimnames'][1:], atts=v['atts'])

jobs.sort()
need = sum(n for _, n, _ in jobs) / 1e9
free = shutil.disk_usage('/content').free / 1e9
big = [k for k in meta['vars'] if '_L' in k]
print(f'\nKeeping {len(jobs)} pieces, {need:.1f} GB; free disk {free:.0f} GB')
print('Large pieces kept:', big)
if len(big) < len(TARGETS) * len(KEEP_LAYERS):
    raise SystemExit('Did not find all target variables. Send me this.')
if need > free - 5:
    raise SystemExit('Not enough disk. Send me this.')

# ---------- one sequential pass through the compressed file ----------
t0 = time.time()
pos, nxt = 0, 10e9
with zf.open(name) as f:
    for start, n, fn in jobs:
        while pos < start:
            got = len(f.read(min(CHUNK, start - pos)))
            if got == 0:
                raise SystemExit('Unexpected end of file')
            pos += got
            if pos >= nxt:
                print(f'  {pos / 1e9:.0f} of {total / 1e9:.0f} GB, '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
                nxt += 10e9
        with open(fn, 'ab') as o:
            left = n
            while left:
                buf = f.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit('Unexpected end of file')
                o.write(buf)
                left -= len(buf)
                pos += len(buf)
                if pos >= nxt:
                    print(f'  {pos / 1e9:.0f} of {total / 1e9:.0f} GB, '
                          f'{(time.time() - t0) / 60:.1f} min', flush=True)
                    nxt += 10e9

with open(f'{OUT}/meta.json', 'w') as o:
    json.dump(meta, o, default=str)
print(f'\nDone in {(time.time() - t0) / 60:.1f} min. Files:')
for fn in sorted(os.listdir(OUT)):
    print(f'  {fn}  {os.path.getsize(f"{OUT}/{fn}") / 1e9:.2f} GB')

# quick sanity check on one high-stack row
k = [x for x in big if x.endswith('_L2')][0]
m = meta['vars'][k]
arr = np.memmap(m['file'], dtype=m['dtype'], mode='r',
                shape=tuple(m['shape']))
row = np.asarray(arr[len(arr) // 2])
print(f'\n{k} middle source row: max {row.max():.3e}, '
      f'sum {row.sum():.3e}, nonzero {np.count_nonzero(row)}')
