/******************************************************************************
 * extract_timeseries.js
 *
 * Google Earth Engine extraction script for the primary study year (2022).
 * Run in the GEE Code Editor (code.earthengine.google.com).
 * Produces one CSV per country per chunk (5 export tasks per country).
 * Download the Drive folder to data/gee_exports/ before running
 * extraction/assemble_sequences.py.
 *
 * Change EE_PROJECT to your own GEE project ID.
 * Change COUNTRIES asset paths to match your uploaded parcel assets.
 *
 * Feature set per monthly composite (14 bands):
 *   Sentinel-2: B5, B6, B7, B11, B12, NDVI, EVI, NDWI, NDMI, s2_valid_obs
 *   Sentinel-1: VV, VH, VH_VV, s1_valid_obs
 *   13 months × 14 features = 182 columns per parcel (plus metadata).
 ******************************************************************************/

var EE_PROJECT    = 'your-gee-project-id';   // ← change this
var YEAR          = 2022;
var SEASON_START  = (YEAR - 1) + '-10-01';   // Oct 2021
var SEASON_END    = YEAR + '-10-31';          // Oct 2022

var S1_ORBIT_PASS = 'DESCENDING';
var ZONAL_SCALE   = 10;
var TILE_SCALE    = 8;
var INWARD_BUFFER_M = -10;
var ID_PROP       = 'field_id';

var COUNTRIES = {
  FR: 'projects/' + EE_PROJECT + '/assets/FR_parcels_sampled',
  ES: 'projects/' + EE_PROJECT + '/assets/ES_parcels_sampled'
};

var OUTPUT_DRIVE_FOLDER = 'parcel_timeseries_export';

// ── Sentinel-2 ───────────────────────────────────────────────────────────────

function maskS2Clouds(image) {
  var scl  = image.select('SCL');
  var good = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(good).divide(10000)
    .copyProperties(image, ['system:time_start']);
}

function addS2Indices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var evi  = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    { NIR: image.select('B8'), RED: image.select('B4'), BLUE: image.select('B2') }
  ).rename('EVI');
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  return image.addBands([ndvi, ndwi, evi, ndmi])
    .copyProperties(image, ['system:time_start']);
}

function getS2Collection(aoi) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(SEASON_START, SEASON_END)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 70))
    .map(maskS2Clouds)
    .map(addS2Indices)
    .select(['B5', 'B6', 'B7', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI', 'NDMI']);
}

// ── Sentinel-1 ───────────────────────────────────────────────────────────────

function speckleFilter(image) {
  return image.focalMedian({radius: 30, units: 'meters'})
    .copyProperties(image, image.propertyNames());
}

function addVhVvRatio(image) {
  return image.addBands(
    image.select('VH').subtract(image.select('VV')).rename('VH_VV')
  ).copyProperties(image, ['system:time_start']);
}

function getS1Collection(aoi) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(SEASON_START, SEASON_END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', S1_ORBIT_PASS))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .map(speckleFilter)
    .map(addVhVvRatio)
    .select(['VV', 'VH', 'VH_VV']);
}

// ── Monthly compositing ──────────────────────────────────────────────────────

function generateMonthlyDateStrings(startStr, endStr) {
  var start = new Date(startStr), end = new Date(endStr), dates = [];
  var cur   = new Date(start.getFullYear(), start.getMonth(), 1);
  while (cur < end) {
    var yyyy = cur.getFullYear(), mm = ('0' + (cur.getMonth() + 1)).slice(-2);
    dates.push('' + yyyy + mm + '01');
    cur.setMonth(cur.getMonth() + 1);
  }
  return dates;
}

var DATE_STRINGS = generateMonthlyDateStrings(SEASON_START, SEASON_END);

function compositeForWindow(coll, wStart, wEnd, bands, dateStr, prefix) {
  var template = ee.Image.constant(bands.map(function(){ return 0; }))
    .rename(bands).updateMask(ee.Image(0)).float();
  var window   = coll.filterDate(wStart, wEnd);
  var comp     = ee.ImageCollection([template]).merge(window).median()
    .rename(bands.map(function(b){ return b + '_' + dateStr; }));
  var count    = window.select(bands[0]).count()
    .rename(prefix + '_valid_obs_' + dateStr);
  return comp.addBands(count);
}

var CHUNK_SIZE = 3;
function chunkArray(arr, size) {
  var chunks = [];
  for (var i = 0; i < arr.length; i += size) chunks.push(arr.slice(i, i + size));
  return chunks;
}
var DATE_CHUNKS  = chunkArray(DATE_STRINGS, CHUNK_SIZE);
var S2_BANDS     = ['B5', 'B6', 'B7', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI', 'NDMI'];
var S1_BANDS     = ['VV', 'VH', 'VH_VV'];

function buildChunkComposite(aoi, dates) {
  var s2 = getS2Collection(aoi), s1 = getS1Collection(aoi), imgs = [];
  dates.forEach(function(d) {
    var y = parseInt(d.slice(0, 4)), m = parseInt(d.slice(4, 6));
    var wStart = ee.Date.fromYMD(y, m, 1), wEnd = wStart.advance(1, 'month');
    imgs.push(compositeForWindow(s2, wStart, wEnd, S2_BANDS, d, 's2'));
    imgs.push(compositeForWindow(s1, wStart, wEnd, S1_BANDS, d, 's1'));
  });
  return ee.Image.cat(imgs);
}

function predictCols(dates) {
  var cols = [];
  dates.forEach(function(d) {
    S2_BANDS.forEach(function(b){ cols.push(b + '_' + d); });
    cols.push('s2_valid_obs_' + d);
    S1_BANDS.forEach(function(b){ cols.push(b + '_' + d); });
    cols.push('s1_valid_obs_' + d);
  });
  return cols;
}

// ── Export ───────────────────────────────────────────────────────────────────

function processCountry(cc, assetPath) {
  var parcels = ee.FeatureCollection(assetPath).map(function(f) {
    return f.setGeometry(f.geometry().buffer(INWARD_BUFFER_M));
  });
  var aoi = parcels.geometry().convexHull();

  DATE_CHUNKS.forEach(function(dates, idx) {
    var comp  = buildChunkComposite(aoi, dates);
    var stats = comp.reduceRegions({
      collection: parcels, reducer: ee.Reducer.mean(),
      scale: ZONAL_SCALE, tileScale: TILE_SCALE
    }).map(function(f) {
      return f.set('area_ha', ee.Number(f.geometry().area()).divide(10000));
    });

    var taskName = cc + '_chunk' + idx;
    Export.table.toDrive({
      collection: stats,
      description: taskName + '_drive',
      folder: OUTPUT_DRIVE_FOLDER,
      fileNamePrefix: taskName,
      fileFormat: 'CSV',
      selectors: [ID_PROP, 'country', 'crop_class', 'area_ha'].concat(predictCols(dates))
    });
    print(cc + ' chunk ' + idx + ': export task created -- ' + taskName);
  });
}

Object.keys(COUNTRIES).forEach(function(cc) {
  processCountry(cc, COUNTRIES[cc]);
});
