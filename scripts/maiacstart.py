# Start Earth Engine exports: daily MAIAC aerosol optical depth (MODIS
# MCD19A2, 1 km, best-quality retrievals) in 8 wind sectors (2-30 km)
# around NEVER-EXIT coal plants that are at least 60 km from every plant
# with an exit, 2004-2019. Exports run on Google's servers and land in
# MyDrive/maiac_exports. No data near exit plants is requested.
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd, ee
try:
    ee.Initialize(project='coal-closures-air')
except Exception:
    ee.Authenticate(); ee.Initialize(project='coal-closures-air')

def km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    x = (np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2)
         * np.sin((o2 - o1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(x))
fc = pd.read_csv(f'{DIR}/isrm_coal_facility_cells.csv') \
       .dropna(subset=['latitude', 'longitude'])
a = pd.read_csv(f'{DIR}/us_epa_annual_all.csv', low_memory=False,
                usecols=['facilityid', 'year', 'heatinput',
                         'primaryfuelinfo'])
a['heatinput'] = pd.to_numeric(a['heatinput'], errors='coerce').fillna(0)
cy = a[a['primaryfuelinfo'].fillna('').str.contains('Coal')
       & (a['heatinput'] > 0)].groupby('facilityid')['year'].apply(set)
U = pd.read_csv(f'{DIR}/us_unit_exits.csv')
exit_fids = set(U['uid'].str.split('_').str[0].astype(int))
ex = fc[fc['facilityid'].isin(exit_fids)]
ctrl = fc[fc['facilityid'].isin([f for f, ys in cy.items()
                                 if set(range(2003, 2020)) <= ys
                                 and f not in exit_fids])].copy()
ctrl['d_exit'] = [km(ex.latitude.values, ex.longitude.values, r.latitude,
                     r.longitude).min() for r in ctrl.itertuples()]
use = ctrl[ctrl['d_exit'] >= 60]
print('Never-exit plants >= 60 km from any exit plant:', len(use))
use[['facilityid', 'latitude', 'longitude']].to_csv(
    f'{DIR}/maiac_control_plants.csv', index=False)

def sectors(fid, lat, lon):
    feats = []
    p = ee.Geometry.Point([lon, lat])
    ring = p.buffer(30000).difference(p.buffer(2000))
    for k in range(8):
        a0 = np.radians(k * 45 - 22.5); a1 = np.radians(k * 45 + 22.5)
        # wedge as a polygon reaching 40 km, intersected with the ring
        dlat = 40 / 111.0; dlon = 40 / (111.0 * np.cos(np.radians(lat)))
        pts = [[lon, lat]] + [[lon + dlon * np.sin(t), lat + dlat * np.cos(t)]
                              for t in np.linspace(a0, a1, 7)] + [[lon, lat]]
        wedge = ee.Geometry.Polygon([pts])
        feats.append(ee.Feature(ring.intersection(wedge, 100),
                                {'fid': int(fid), 'sector': k}))
    return feats
F = ee.FeatureCollection([f for r in use.itertuples()
                          for f in sectors(r.facilityid, r.latitude,
                                           r.longitude)])

col = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
def day_img(d):
    d = ee.Date(d)
    ic = col.filterDate(d, d.advance(1, 'day'))
    def qa(img):
        best = img.select('AOD_QA').rightShift(8).bitwiseAnd(15).eq(0)
        return img.select('Optical_Depth_055').multiply(0.001) \
                  .updateMask(best)
    empty = ee.Image.constant(0).rename('aod').updateMask(
        ee.Image.constant(0))
    img = ee.Image(ee.Algorithms.If(ic.size().gt(0),
                                    ic.map(qa).mean().rename('aod'), empty))
    return img.set('date', d.format('YYYY-MM-dd'))

for y in range(2004, 2020):
    start = ee.Date(f'{y}-01-01')
    n = ee.Date(f'{y + 1}-01-01').difference(start, 'day')
    days = ee.List.sequence(0, n.subtract(1)).map(
        lambda i: start.advance(i, 'day'))
    def per_day(d):
        img = day_img(d)
        red = img.reduceRegions(F, ee.Reducer.mean().combine(
            ee.Reducer.count(), sharedInputs=True), 1000)
        return red.map(lambda f: f.set('date', ee.Date(d)
                                       .format('YYYY-MM-dd'))
                       .setGeometry(None))
    table = ee.FeatureCollection(days.map(per_day)).flatten() \
              .filter(ee.Filter.notNull(['mean']))
    task = ee.batch.Export.table.toDrive(
        collection=table, description=f'maiac_controls_{y}',
        folder='maiac_exports', fileNamePrefix=f'maiac_controls_{y}',
        selectors=['fid', 'sector', 'date', 'mean', 'count'])
    task.start()
    print('Started export', y)
print('\nExports run on Google servers. Check progress with:\n'
      '  import ee; [print(t.config["description"], t.state) for t in '
      'ee.batch.Task.list()[:16]]')
