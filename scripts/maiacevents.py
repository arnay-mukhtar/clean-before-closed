# POST-REGISTRATION: start Earth Engine exports of daily MAIAC aerosol
# optical depth in 8 wind sectors (2-30 km) around the EXIT plants of the
# H3f sample (SO2 drop >= 5,000 t, events 2005-2018), all years 2004-2019
# (the event years plus the years needed for placebo dates at these
# plants). Same retrieval, QA and geometry as maiacstart.py (controls).
# Exports land in MyDrive/maiac_exports as maiac_events_v2_<year>_<batch>.csv
# (v2: the first run failed because the mapped function had two arguments).
DIR = '/content/drive/MyDrive/coal_closures'
import numpy as np, pandas as pd, ee
try:
    ee.Initialize(project='coal-closures-air')
except Exception:
    ee.Authenticate(); ee.Initialize(project='coal-closures-air')

E = pd.read_csv(f'{DIR}/us_plant_exit_events.csv')
ev = E[(E['so2_drop'] >= 5000) & E['event_year'].between(2005, 2018)]
print('H3f events:', len(ev), ' plants:', ev['fid'].nunique())
pl = ev.drop_duplicates('fid')[['fid', 'lat', 'lon']]
pl.to_csv(f'{DIR}/maiac_event_plants.csv', index=False)

def sectors(fid, lat, lon):
    feats = []
    p = ee.Geometry.Point([lon, lat])
    ring = p.buffer(30000).difference(p.buffer(2000))
    for k in range(8):
        a0 = np.radians(k * 45 - 22.5); a1 = np.radians(k * 45 + 22.5)
        dlat = 40 / 111.0; dlon = 40 / (111.0 * np.cos(np.radians(lat)))
        pts = [[lon, lat]] + [[lon + dlon * np.sin(t), lat + dlat * np.cos(t)]
                              for t in np.linspace(a0, a1, 7)] + [[lon, lat]]
        wedge = ee.Geometry.Polygon([pts])
        feats.append(ee.Feature(ring.intersection(wedge, 100),
                                {'fid': int(fid), 'sector': k}))
    return feats

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

def make_per_day(F):
    # Earth Engine maps a function with exactly one argument; the feature
    # collection is captured by closure (a default argument breaks the map)
    def per_day(d):
        img = day_img(d)
        red = img.reduceRegions(F, ee.Reducer.mean().combine(
            ee.Reducer.count(), sharedInputs=True), 1000)
        return red.map(lambda f: f.set('date', ee.Date(d)
                                       .format('YYYY-MM-dd'))
                       .setGeometry(None))
    return per_day

batches = [pl.iloc[i:i + 45] for i in range(0, len(pl), 45)]
for bi, bp in enumerate(batches):
    F = ee.FeatureCollection([f for r in bp.itertuples()
                              for f in sectors(r.fid, r.lat, r.lon)])
    for y in range(2004, 2020):
        start = ee.Date(f'{y}-01-01')
        n = ee.Date(f'{y + 1}-01-01').difference(start, 'day')
        days = ee.List.sequence(0, n.subtract(1)).map(
            lambda i: start.advance(i, 'day'))
        table = ee.FeatureCollection(days.map(make_per_day(F))).flatten() \
                  .filter(ee.Filter.notNull(['mean']))
        task = ee.batch.Export.table.toDrive(
            collection=table, description=f'maiac_events_v2_{y}_{bi}',
            folder='maiac_exports', fileNamePrefix=f'maiac_events_v2_{y}_{bi}',
            selectors=['fid', 'sector', 'date', 'mean', 'count'])
        task.start()
        print('Started export', y, 'batch', bi, flush=True)
print(f'\n{16 * len(batches)} exports running on Google servers. Check with:'
      '\n  import ee; [print(t.config["description"], t.state) for t in '
      'ee.batch.Task.list()[:40]]')
