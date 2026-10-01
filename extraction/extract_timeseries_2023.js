/**
 * extraction/extract_timeseries_2023.js
 *
 * Google Earth Engine script: Sentinel-1/2 parcel time-series extraction,
 * Spain 2023 (drought robustness experiment, §3.8).
 *
 * Identical to extract_timeseries.js except:
 *   - YEAR = 2023 (season: Mar 2023 – Mar 2024)
 *   - Only Spain is extracted (France data not needed for this experiment)
 *   - Output folder: parcel_timeseries_export_2023
 *
 * The same parcel footprint (ES_parcels asset) is re-used from the 2022
 * study.  Parcel selection criteria remain unchanged.
 *
 * See extract_timeseries.js for full documentation of the extraction
 * methodology (cloud masking, speckle filtering, composite scheme, etc.).
 */

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------

var EE_PROJECT = 'your-gcp-project-id';

var PARCEL_ASSETS = {
  ES: 'users/<you>/ES_parcels',
};

var OUTPUT_DRIVE_FOLDER = 'parcel_timeseries_export_2023';

var YEAR          = 2023;
var S1_ORBIT_PASS = 'DESCENDING';
var ZONAL_SCALE   = 10;
var TILE_SCALE    = 4;
var BUFFER_M      = -10;
var N_CHUNKS      = 5;


// ---------------------------------------------------------------------------
// Sentinel-2
// ---------------------------------------------------------------------------

function maskS2Clouds(image) {
  var scl  = image.select('SCL');
  var good = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(good);
}

function addS2Indices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndmi = image.normalizedDifference(['B8A', 'B11']).rename('NDMI');
  var evi  = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    { NIR: image.select('B8'), RED: image.select('B4'), BLUE: image.select('B2') }
  ).rename('EVI');
  return image.addBands([ndvi, ndwi, ndmi, evi]);
}

function getS2Collection(aoi, start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .map(maskS2Clouds)
    .map(addS2Indices)
    .select(['B5', 'B6', 'B7', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI', 'NDMI']);
}


// ---------------------------------------------------------------------------
// Sentinel-1
// ---------------------------------------------------------------------------

function speckleFilter(image) {
  return image.focalMedian(30, 'circle', 'meters')
    .copyProperties(image, image.propertyNames());
}

function addVhVvRatio(image) {
  return image.addBands(
    image.select('VH').subtract(image.select('VV')).rename('VH_VV')
  );
}

function getS1Collection(aoi, start, end) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', S1_ORBIT_PASS))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .map(speckleFilter)
    .map(addVhVvRatio)
    .select(['VV', 'VH', 'VH_VV']);
}


// ---------------------------------------------------------------------------
// Monthly composites
// ---------------------------------------------------------------------------

function monthlyComposite(collection, year, month, bands) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end   = start.advance(1, 'month');
  var win   = collection.filterDate(start, end);
  var obs   = win.select(bands[0]).count().rename('obs_count');
  return win.median().addBands(obs).set('composite_date',
    start.format('YYYYMMdd'));
}


// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

function extractChunk(country, chunkIndex, totalChunks) {
  var parcels = ee.FeatureCollection(PARCEL_ASSETS[country]);
  var n       = parcels.size().getInfo();
  var chunk   = Math.ceil(n / totalChunks);
  var start   = chunkIndex * chunk;
  var subset  = ee.FeatureCollection(
    parcels.toList(chunk, start).map(function(f) { return ee.Feature(f); })
  );

  var aoi = subset.geometry().bounds();

  var s2 = getS2Collection(aoi, YEAR + '-01-01', (YEAR + 1) + '-04-01');
  var s1 = getS1Collection(aoi, YEAR + '-01-01', (YEAR + 1) + '-04-01');

  var compositeYearMonths = [
    [YEAR, 3], [YEAR, 4], [YEAR, 5], [YEAR, 6], [YEAR, 7],
    [YEAR, 8], [YEAR, 9], [YEAR, 10], [YEAR, 11], [YEAR, 12],
    [YEAR + 1, 1], [YEAR + 1, 2], [YEAR + 1, 3]
  ];

  var s2Bands = ['B5', 'B6', 'B7', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI', 'NDMI'];
  var s1Bands = ['VV', 'VH', 'VH_VV'];

  var stackedImage = ee.Image([]);

  compositeYearMonths.forEach(function(ym) {
    var yr = ym[0]; var mo = ym[1];
    var dateStr = ee.Date.fromYMD(yr, mo, 1).format('YYYYMMdd').getInfo();

    var s2comp = monthlyComposite(s2, yr, mo, s2Bands);
    var s1comp = monthlyComposite(s1, yr, mo, s1Bands);

    s2Bands.forEach(function(b) {
      stackedImage = stackedImage.addBands(
        s2comp.select(b).rename(b + '_' + dateStr)
      );
    });
    stackedImage = stackedImage.addBands(
      s2comp.select('obs_count').rename('s2_valid_obs_' + dateStr)
    );
    s1Bands.forEach(function(b) {
      stackedImage = stackedImage.addBands(
        s1comp.select(b).rename(b + '_' + dateStr)
      );
    });
    stackedImage = stackedImage.addBands(
      s1comp.select('obs_count').rename('s1_valid_obs_' + dateStr)
    );
  });

  var bufferedSubset = subset.map(function(f) { return f.buffer(BUFFER_M); });

  var stats = stackedImage.reduceRegions({
    collection: bufferedSubset,
    reducer:    ee.Reducer.mean(),
    scale:      ZONAL_SCALE,
    tileScale:  TILE_SCALE,
  }).map(function(f) { return f.set({ country: country }); });

  var taskName = country + '_chunk' + chunkIndex;
  Export.table.toDrive({
    collection:  stats,
    description: taskName,
    folder:      OUTPUT_DRIVE_FOLDER,
    fileFormat:  'CSV',
  });

  print('Submitted: ' + taskName + ' (2023)');
}


// ---------------------------------------------------------------------------
// Run (Spain only)
// ---------------------------------------------------------------------------

for (var i = 0; i < N_CHUNKS; i++) {
  extractChunk('ES', i, N_CHUNKS);
}
