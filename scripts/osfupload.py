# Upload the deposit to your OSF project's storage, into folders data/,
# data/step1/ and scripts/, plus fingerprints.csv at the top level.
# Needs two Colab Secrets: OSF_TOKEN (personal access token with
# osf.full_write) and OSF_PROJECT (the 5-character code in the project URL,
# e.g. abcde from osf.io/abcde/).
DIR = '/content/drive/MyDrive/coal_closures'
import os, glob, requests
from google.colab import userdata

TOKEN = userdata.get('OSF_TOKEN')
PID = userdata.get('OSF_PROJECT').strip().strip('/').split('/')[-1]
H = {'Authorization': f'Bearer {TOKEN}'}
API = 'https://api.osf.io/v2'

DATA = ['final_pm25_35km_predictions.csv', 'final_pm25_35km_placebo.csv',
        'final_so4_100km_predictions.csv', 'final_so4_100km_placebo.csv',
        'final_so4_50km_predictions.csv', 'final_so4_50km_placebo.csv',
        'maiac_control_plants.csv', 'maiac_control_plantyears.csv',
        'maiac_placebo_draws.csv', 'so2_check_final.csv',
        'model_predictions_us_events_v2.csv',
        'model_predictions_us_events.csv', 'model_predictions_us.csv',
        'model_predictions_us_monitors.csv', 'us_epa_annual_all.csv',
        'us_plant_exit_events.csv', 'us_unit_exits.csv',
        'us_unit_exits_damages.csv', 'us_unit_exit_trajectories.csv',
        'counterfactual_paths.csv', 'us_closure_dates.csv',
        'eu_unit_exits.csv', 'eu_coal_sites.csv', 'eu_station_coverage.csv',
        'isrm_coal_facility_cells.csv', 'design_tuning.csv',
        'hourly_event_coverage.csv', 'aqs_hourly_monitor_years.csv',
        'downwind_placebo_draws.csv', 'downwind_hourly_placebo_draws.csv',
        'wind_daily_plants.csv', 'so4_event_coverage.csv',
        'so4_primary_events.csv', 'so4_placebo_draws.csv',
        'so2_check_coverage.csv', 'placebo_draws_2004_2018.csv']
STEP1 = ['nei_pm25_by_oris_v3.csv', 'eia860_stacks.csv']
SCRIPTS = sorted(os.path.basename(p) for p in glob.glob(f'{DIR}/*.py')
                 if '(1)' not in p)

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
    name = os.path.basename(path)
    if name in existing:
        print('  already there:', name); return
    with open(path, 'rb') as f:
        r = requests.put(join(up_link, f'kind=file&name={name}'), data=f,
                         headers=H, timeout=3600)
    check(r, f'Uploading {name}')
    print('  uploaded:', name, f'({os.path.getsize(path) / 1e6:.1f} MB)',
          flush=True)

root_files = osf['relationships']['files']['links']['related']['href']
d_up, d_new, d_files = folder(ROOT_NEW, root_files, 'data')
s_up, s_new, s_files = folder(ROOT_NEW, root_files, 'scripts')
st_up, _, st_files = folder(d_new, d_files, 'step1')

print('Top level:')
upload(ROOT_UP, f'{DIR}/fingerprints.csv', list_children(root_files))
print('data/:')
have = list_children(d_files)
for f in DATA:
    p = f'{DIR}/{f}'
    if os.path.exists(p):
        upload(d_up, p, have)
    else:
        print('  MISSING on Drive:', f)
print('data/step1/:')
have = list_children(st_files)
for f in STEP1:
    upload(st_up, f'{DIR}/step1/{f}', have)
print('scripts/:')
have = list_children(s_files)
for f in SCRIPTS:
    upload(s_up, f'{DIR}/{f}', have)
print(f'\nDone: fingerprints.csv, {len(DATA) + len(STEP1)} data files, '
      f'{len(SCRIPTS)} scripts. Upload the pre-registration PDF by hand.')
