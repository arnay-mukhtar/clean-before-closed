# Post-registration deposit (10 Nov 2026; version 2, 15:50: files whose
# content changed are uploaded as new OSF versions, checked by SHA-256, and
# batch A's EIA-860 generator panel and EPA-EIA crosswalk are added). Uploads, into a new folder
# post_registration/ in the OSF project's storage (the registrations stay
# frozen): every analysis script as it now stands (scripts/), every output
# in coal_closures/outcomes (outcomes/), the derived data the new analyses
# use (data/), and a SHA-256 manifest of all of them
# (post_registration_fingerprints.csv).
# Needs two Colab Secrets: OSF_TOKEN (personal access token with
# osf.full_write) and OSF_PROJECT (the 5-character code in the project URL,
# e.g. abcde from osf.io/abcde/).
DIR = '/content/drive/MyDrive/coal_closures'
import os, glob, hashlib, requests
from google.colab import userdata

TOKEN = userdata.get('OSF_TOKEN')
PID = userdata.get('OSF_PROJECT').strip().strip('/').split('/')[-1]
H = {'Authorization': f'Bearer {TOKEN}'}
API = 'https://api.osf.io/v2'

def check(r, what):
    if not r.ok:
        raise SystemExit(f'{what} failed: {r.status_code} {r.text[:300]}')
    return r

root = check(requests.get(f'{API}/nodes/{PID}/files/', headers=H),
             'Reading project storage').json()
osf = [p for p in root['data'] if p['attributes']['name'] == 'osfstorage'][0]
ROOT_UP = osf['links']['upload']
ROOT_NEW = osf['links']['new_folder']
print('Connected to project', PID)

def join(link, params):
    return link + ('&' if '?' in link else '?') + params

def list_children(files_link):
    out = {}
    url = files_link
    while url:
        j = check(requests.get(url, headers=H), 'Listing').json()
        for it in j['data']:
            out[it['attributes']['name']] = it
        url = j['links'].get('next')
    return out

def folder(parent_new, parent_files, name):
    kids = list_children(parent_files)
    if name in kids:
        it = kids[name]
    else:
        r = check(requests.put(join(parent_new, f'name={name}'), headers=H),
                  f'Creating folder {name}').json()
        it = r['data']
    up = it['links']['upload']
    new = it['links'].get('new_folder')
    files = it['relationships']['files']['links']['related']['href'] \
        if 'relationships' in it else None
    if files is None:      # WaterButler response: look the folder up again
        it = list_children(parent_files)[name]
        up, new = it['links']['upload'], it['links'].get('new_folder')
        files = it['relationships']['files']['links']['related']['href']
    return up, new, files

def upload(up_link, path, existing):
    """New file: upload. Same name and same SHA-256 on OSF: skip. Same name
    but changed content: upload a new version of that OSF file (OSF keeps
    every earlier version)."""
    name = os.path.basename(path)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if name in existing:
        it = existing[name]
        osf_h = (it['attributes'].get('extra') or {}).get('hashes', {}) \
            .get('sha256')
        if osf_h == h:
            print('  unchanged:', name); return
        with open(path, 'rb') as f:
            r = requests.put(join(it['links']['upload'], 'kind=file'),
                             data=f, headers=H, timeout=3600)
        check(r, f'Updating {name}')
        print('  updated (new version):', name, flush=True); return
    with open(path, 'rb') as f:
        r = requests.put(join(up_link, f'kind=file&name={name}'), data=f,
                         headers=H, timeout=3600)
    check(r, f'Uploading {name}')
    print('  uploaded:', name, f'({os.path.getsize(path) / 1e6:.1f} MB)',
          flush=True)

DATA = ['counterfactual_paths_v2.csv', 'cea_units_long.csv',
        'eia860_plant_panel.csv', 'eia860_retired.csv',
        'us_epa_monthly_closures.csv', 'almaty_prediction_draws.csv',
        'easiur/egrid_2016_plant_easiur.csv',
        'eia860_generator_panel.csv', 'epa_eia_crosswalk.csv',
        'komati_eskom.csv', 'tropomi_no2_annual.csv',
        'pm25_daily_events.csv', 'pm25_hourly_events.csv',
        'wind_hourly_events.csv', 'eea_aq/station_months.csv']
DATA = [f for f in DATA if os.path.exists(f'{DIR}/{f}')]
SCRIPTS = sorted(p for p in glob.glob(f'{DIR}/*.py') if '(1)' not in p)
OUTS = sorted(p for p in glob.glob(f'{DIR}/outcomes/*') if os.path.isfile(p)
              and '(1)' not in p)

root_files = osf['relationships']['files']['links']['related']['href']
pr_up, pr_new, pr_files = folder(ROOT_NEW, root_files, 'post_registration')
groups = []
for name, paths in [('scripts', SCRIPTS), ('outcomes', OUTS),
                    ('data', [f'{DIR}/{f}' for f in DATA])]:
    up, _, files = folder(pr_new, pr_files, name)
    groups.append((name, up, files, paths))

rows = []
for name, up, files, paths in groups:
    print(f'post_registration/{name}/:')
    have = list_children(files)
    for p in paths:
        if not os.path.exists(p):
            print('  MISSING on Drive:', os.path.basename(p)); continue
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        rows.append(f'{name}/{os.path.basename(p)},{h}')
        upload(up, p, have)
man = f'{DIR}/post_registration_fingerprints.csv'
open(man, 'w').write('file,sha256\n' + '\n'.join(rows) + '\n')
upload(pr_up, man, list_children(pr_files))
print(f'\nDone: {len(rows)} files fingerprinted and uploaded to post_registration/.')
