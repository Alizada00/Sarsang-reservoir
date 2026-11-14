// =====================================================================
// sarsang_dataset.js   (VERSION v9)     THE ONLY Earth Engine file
//   v9: the timeseries and blocks modes are the ORIGINAL tested code again (v8 had a rewritten
//       timeseries that failed with 'bucketMeans' because Sentinel-1 had no coverage filter).
//
//   MODE = 'timeseries'  Rough water-area time series for one year. Run once per year,
//                        2020 to present. Output: one Drive CSV per run. Used to find the
//                        8 development dates. The 8 dates then go into DEV_DATES below.
//   MODE = 'blocks'      1 km block split (train / test / buffer). Run ONCE and export as
//                        an asset. After that never run again (CONFIG.BLOCKS_ASSET is frozen).
//   MODE = 'build'       Builds the dataset and exports ONE CSV: sarsang_dataset_<VERSION>.csv
//                        Column "set" tells the rows apart:
//                          set = train : weakly labeled points (train blocks only, filtered)
//                          set = label : points to label by hand (test blocks only, not filtered),
//                                        with stratum_px and frame_px for the area-weighted accuracy
//   MODE = 'label'       Blind manual labeling tool for ONE date. Its output is a small CSV per
//                        date (val_<date>_<annotator>.csv). The Python file merges them.
//
// FULL WORKFLOW FROM SCRATCH
//   1. timeseries  Run year by year (2020-01-01/2021-01-01, then 2021, 2022 ...).
//                  Inspect the CSVs. Choose 8 dates. Write them in DEV_DATES below.
//   2. blocks      Set ASSET_ID. Run once. Export. Never run again.
//   3. build       Run. One export task: sarsang_dataset_v9. Wait until done.
//   4. label       DATE_INDEX 0..7. Run, label, copy CSV from Console, save as val_*.csv.
//   5. sarsang_model.py  RUN_FINAL_TEST = False -> choose model.
//   6. sarsang_model.py  RUN_FINAL_TEST = True  -> final numbers.
//
// VERSIONING: if something is corrected, this SAME file is edited and VERSION goes up
//   (v9 -> v10). The dataset CSV name follows the version.
//
// NOTE: I could not run this in Earth Engine. A mock test only checked that the JavaScript
//       logic has no typos or undefined names.
// =====================================================================

var MODE = 'build';            // 'timeseries' | 'blocks' | 'build' | 'label'

var CONFIG = {
  VERSION: 'v9',
  CRS: 'EPSG:32638',
  ROI_BUFFER_M: 3000,
  POSSIBLE_BUFFER_M: 150,      // "possible water zone" = JRC ever-water + this margin
  CS_MIN: 0.6,                 // Cloud Score+ minimum (Sentinel-2)
  EXPORT_FOLDER: 'sarsang_phd',
  ASSET_FOLDER: '',

  // ---- MODE = 'timeseries' ----
  // Run one year at a time. Change START / END each run. END is exclusive.
  START: '2020-01-01',
  END:   '2021-01-01',
  MIN_VALID_FRAC: 0.90,        // share of the possible water zone that must be cloud-free
  S1_FIXED_DB: -16,            // fixed threshold for the area_fixed column (Sentinel-1)
  USE_S2: true, USE_LANDSAT: true, USE_S1: true,
  EXPORT_NAME: 'sarsang_area_ts',
  SHOW_CHART: false,
  SCALE: 20,
  SCALE_LANDSAT: 30,

  // ---- MODE = 'blocks' (run ONCE, then set ASSET_ID and export) ----
  BLOCK_M: 1000,
  TEST_SHARE: 0.3,
  BUFFER_M: 300,
  ASSET_ID: '',                // set to 'projects/.../assets/sarsang_blocks' before running

  // frozen block split (173 blocks, seed 42, 300 m buffer). Never change.
  BLOCKS_ASSET: '',

  // ---- MODE = 'build' ----
  S1_WATER_DB: -16,            // smoothed VV below this = water-like
  S1_LAND_DB: -12,             // smoothed VV above this = land-like
  MNDWI_WATER: 0.1,
  MNDWI_LAND: -0.1,
  N_LABEL: {aw: 15, al: 15, hard: 50, other: 10},   // 90 per date (used only if sampling anew)
  N_TRAIN: {t2w: 150, t2l: 150, t1w: 40, t1l: 40},  // 380 per date
  WATER_REF_M: 678,            // SRTM water surface of the reservoir (metres)
  SHADOW_ABOVE_M: 60,          // weak water this far above the surface = shadow
  SEED: 42,
  RUN_INDICES: null,           // null = all 8 dates, [0] = first date only (smoke test)
  PREVIEW_INDEX: 0,
  // Label rows must be exactly the points people label. They already live in this asset
  // (made earlier). true = read them from the asset. false = sample them anew (fresh start).
  LABEL_POINTS_FROM_ASSET: true,
  EXPORT_LABEL_ASSET: false,   // true only together with LABEL_POINTS_FROM_ASSET = false
  LABEL_ASSET: '',

  // ---- MODE = 'label' ----
  DATE_INDEX: 0,               // 0..7, one date per session
  ANNOTATOR: 'A1',             // 'A1' main, 'A2' second annotator (kappa)
  SUBSET_SHARE: 1.0,           // A2 uses 0.25
  START_INDEX: 0,              // resume point (points already saved)
  ORDER_SEED: 7,               // same for everybody. Do NOT change
  ZOOM: 18,
  AUTOPRINT_EVERY: 15,

  // the 8 development dates: Sentinel-2 day + trusted Sentinel-1 group (day_orbit_pass)
  DEV_DATES: [
    {d: '2023-04-28', s1: '2023-04-27_79_DESCENDING'},   // 2.70 km2, lowest water
    {d: '2023-03-24', s1: '2023-03-22_79_DESCENDING'},   // 4.10
    {d: '2020-12-27', s1: '2020-12-26_79_DESCENDING'},   // 5.9, winter
    {d: '2023-07-05', s1: '2023-07-08_79_DESCENDING'},   // 6.9, summer
    {d: '2021-11-19', s1: '2021-11-21_174_ASCENDING'},   // 7.5, autumn
    {d: '2024-04-27', s1: '2024-04-27_174_ASCENDING'},   // 8.6
    {d: '2024-11-21', s1: '2024-11-23_79_DESCENDING'},   // 9.1
    {d: '2024-06-21', s1: '2024-06-21_101_ASCENDING'}    // 10.5
  ],
  HOLDOUT_DATES: ['2025-10-06', '2025-02-16', '2026-08-30']   // never used for training
};

// ---------- ROI ----------
var roi = (typeof geometry !== 'undefined') ? geometry :
  ee.Geometry.Polygon([[
    [46.63911588794778, 40.21770073312651],
    [46.55053861743997, 40.14583080036103],
    [46.59242399341653, 40.13533247141022],
    [46.66850375153126, 40.20272454317249],
    [46.63911588794778, 40.21770073312651]
  ]]);
var area = roi.buffer(CONFIG.ROI_BUFFER_M);

// ---------- WATER ZONES ----------
// shoreZone: places where water has EVER been (JRC 1984-2021) + margin, minus permanent water.
// stableWater / landFar: static training classes (they come from JRC, so jrc_occ must never
// be a feature).
var jrc    = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var occ    = jrc.select('occurrence').unmask(0);
var maxExt = jrc.select('max_extent').unmask(0);
var possibleWater = maxExt.gt(0).focal_max({radius: CONFIG.POSSIBLE_BUFFER_M, units: 'meters'});
var shoreZone   = possibleWater.and(occ.lt(95));
var stableWater = occ.gte(95).focal_min({radius: 45, units: 'meters'});
var landFar     = maxExt.eq(0).and(maxExt.gt(0).focal_max({radius: 500, units: 'meters'}).not());

var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var dem = ee.Image('USGS/SRTMGL1_003');

// Block roles: 0 train, 1 test, 2 buffer.
function roleImage(blocksFC) {
  var testGeom = blocksFC.filter(ee.Filter.eq('role', 'test')).geometry();
  var bufGeom  = testGeom.buffer(CONFIG.BUFFER_M);
  return ee.Image(0)
      .paint(ee.FeatureCollection([ee.Feature(bufGeom)]), 2)
      .paint(ee.FeatureCollection([ee.Feature(testGeom)]), 1)
      .rename('role');
}

// =====================================================================
// MODE = 'timeseries'   ROUGH WATER-AREA TIME SERIES  (the original tested code)
// =====================================================================
// Otsu: automatic threshold between two histogram peaks (split points 1 .. size-1).
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means  = ee.Array(histogram.get('bucketMeans'));
  var size   = ee.Number(means.length().get([0]));
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum    = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean   = sum.divide(total);
  var indices = ee.List.sequence(1, size.subtract(1));
  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = ee.Number(aCounts.reduce(ee.Reducer.sum(), [0]).get([0]));
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0])
                        .divide(aCount.max(1e-9));
    var bCount  = ee.Number(total).subtract(aCount);
    var bMean   = ee.Number(sum).subtract(aCount.multiply(aMean)).divide(bCount.max(1e-9));
    return aCount.multiply(aMean.subtract(mean).pow(2)).add(
           bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  var candidates = means.slice(0, 0, size.subtract(1));
  return candidates.sort(bss).get([-1]);
}

function makeTools() {
  // Every image gets a 'key'. Optical sensors: key = day. Sentinel-1: day_orbit_pass.
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi).filterDate(CONFIG.START, CONFIG.END)
      .linkCollection(csp, ['cs_cdf'])
      .map(function(i) {
        var day = i.date().format('YYYY-MM-dd');
        return ee.Image.cat([
              i.select('B3').divide(10000).rename('G'),
              i.select('B11').divide(10000).rename('S'),
              i.select('SCL').eq(11).rename('snowqa')])
            .updateMask(i.select('cs_cdf').gte(CONFIG.CS_MIN))
            .set({key: day, sid: i.get('system:index')})
            .copyProperties(i, ['system:time_start']);
      });

  // Landsat C2 L2. QA_PIXEL bits 0-4 = fill/cloud/shadow (31), bit 5 = snow (32).
  // QA_RADSAT: bands 3 and 6 saturation (4 + 32 = 36).
  var ls = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
      .filterBounds(roi).filterDate(CONFIG.START, CONFIG.END)
      .map(function(i) {
        var qa  = i.select('QA_PIXEL');
        var sat = i.select('QA_RADSAT');
        var ok = qa.bitwiseAnd(31).eq(0).and(sat.bitwiseAnd(36).eq(0)).and(i.select('SR_B3').gt(0));
        var day = i.date().format('YYYY-MM-dd');
        return ee.Image.cat([
              i.select('SR_B3').multiply(0.0000275).add(-0.2).rename('G'),
              i.select('SR_B6').multiply(0.0000275).add(-0.2).rename('S'),
              qa.bitwiseAnd(32).neq(0).rename('snowqa')])
            .updateMask(ok)
            .set({key: day, sid: i.get('system:index')})
            .copyProperties(i, ['system:time_start']);
      });

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(roi).filterDate(CONFIG.START, CONFIG.END)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .map(function(i) {
        var key = i.date().format('YYYY-MM-dd').cat('_')
            .cat(ee.Number(i.get('relativeOrbitNumber_start')).format('%d')).cat('_')
            .cat(ee.String(i.get('orbitProperties_pass')));
        return i.set({key: key, sid: i.get('system:index')});
      });

  function dayCol(col, key) {
    key = ee.String(key);
    var t0 = ee.Date(key.split('_').get(0));
    return col.filterDate(t0, t0.advance(1, 'day')).filter(ee.Filter.eq('key', key));
  }

  // Keeps only the keys whose scenes cover at least MIN_VALID_FRAC of the possible water
  // zone. This is what keeps the Otsu histograms non-empty. Used for ALL three sensors.
  function goodKeysOf(col, band, scale) {
    var keys = col.aggregate_array('key').distinct().sort();
    var fc = ee.FeatureCollection(keys.map(function(k) {
      var vf = dayCol(col, k).mosaic().select(band).mask().unmask(0).updateMask(possibleWater).reduceRegion({
        reducer: ee.Reducer.mean(), geometry: roi, scale: scale,
        crs: CONFIG.CRS, maxPixels: 1e9, tileScale: 4
      }).get(band);
      return ee.Feature(null, {key: k, valid_frac: vf});
    }));
    return fc.filter(ee.Filter.gte('valid_frac', CONFIG.MIN_VALID_FRAC)).aggregate_array('key');
  }

  function sumArea(stack, scale) {
    return stack.reduceRegion({
      reducer: ee.Reducer.sum(), geometry: roi, scale: scale,
      crs: CONFIG.CRS, maxPixels: 1e9, tileScale: 4
    });
  }

  function optClassify(col, key, scale) {
    var dc  = dayCol(col, key);
    var img = dc.mosaic();
    var g = img.select('G').max(0), s = img.select('S').max(0);
    var mndwi = g.subtract(s).divide(g.add(s).max(0.0001)).rename('mndwi');
    // Snow warning only: MNDWI(Green, SWIR1) is the same formula as NDSI.
    var snow = img.select('snowqa').or(mndwi.gt(0.4).and(g.gt(0.25))).rename('snow');
    // Otsu histogram only inside the possible water zone (balanced water/land mix).
    var hist = mndwi.updateMask(possibleWater).reduceRegion({
      reducer: ee.Reducer.histogram(255, 0.01), geometry: roi, scale: scale,
      crs: CONFIG.CRS, maxPixels: 1e9, tileScale: 4
    }).get('mndwi');
    var thr = ee.Number(otsu(hist));
    return {dc: dc, g: g, mndwi: mndwi, snow: snow, thr: thr, water: mndwi.gt(thr)};
  }

  function optRow(col, sensor, scale) {
    return function(key) {
      key = ee.String(key);
      var d = ee.String(key.split('_').get(0));
      var C = optClassify(col, key, scale);
      var pa = ee.Image.pixelArea().divide(1e6);
      var best = C.water.and(C.snow.not()).and(possibleWater);
      var sums = sumArea(ee.Image.cat([
        C.water.multiply(pa).rename('area_otsu'),
        best.multiply(pa).rename('area_best'),
        C.mndwi.gt(0).multiply(pa).rename('area_fixed'),
        C.snow.multiply(pa).rename('snow_km2'),
        C.water.and(possibleWater.not()).multiply(pa).rename('out_zone_km2'),
        pa.updateMask(C.g.mask()).rename('valid_km2'),
        pa.updateMask(possibleWater).rename('zone_km2'),
        pa.updateMask(possibleWater).updateMask(C.g.mask()).rename('valid_zone_km2'),
        pa.rename('roi_km2')
      ]), scale);
      return ee.Feature(null, sums).set({
        day: d, date_ms: ee.Date(d).millis(),
        year: ee.Date(d).get('year'), month: ee.Date(d).get('month'),
        sensor: sensor, thr: C.thr,
        n_scenes: C.dc.size(),
        scene_ids: C.dc.aggregate_array('sid').join(';'),
        ver: CONFIG.VERSION
      });
    };
  }

  // Sentinel-1: supporting sensor only. Radar shadow INSIDE the zone is not removed.
  function s1Classify(key) {
    var dc = dayCol(s1, key);
    var vv = dc.select('VV').mosaic().focal_median(30, 'circle', 'meters').rename('vv');
    var hist = vv.updateMask(possibleWater).reduceRegion({
      reducer: ee.Reducer.histogram(255, 0.1), geometry: roi, scale: CONFIG.SCALE,
      crs: CONFIG.CRS, maxPixels: 1e9, tileScale: 4
    }).get('vv');
    var thr = ee.Number(otsu(hist));
    return {dc: dc, vv: vv, thr: thr, water: vv.lt(thr)};
  }

  function s1Row(key) {
    key = ee.String(key);
    var d = ee.String(key.split('_').get(0));
    var C = s1Classify(key);
    var pa = ee.Image.pixelArea().divide(1e6);
    var sums = sumArea(ee.Image.cat([
      C.water.multiply(pa).rename('area_otsu'),
      C.water.and(possibleWater).multiply(pa).rename('area_best'),
      C.vv.lt(CONFIG.S1_FIXED_DB).multiply(pa).rename('area_fixed'),
      C.water.and(possibleWater.not()).multiply(pa).rename('out_zone_km2'),
      pa.updateMask(C.vv.mask()).rename('valid_km2'),
      pa.updateMask(possibleWater).rename('zone_km2'),
      pa.updateMask(possibleWater).updateMask(C.vv.mask()).rename('valid_zone_km2'),
      pa.rename('roi_km2')
    ]), CONFIG.SCALE);
    var first = C.dc.first();
    return ee.Feature(null, sums).set({
      day: d, date_ms: ee.Date(d).millis(),
      year: ee.Date(d).get('year'), month: ee.Date(d).get('month'),
      sensor: 'S1', thr: C.thr,
      n_scenes: C.dc.size(),
      scene_ids: C.dc.aggregate_array('sid').join(';'),
      orbit: first.get('relativeOrbitNumber_start'),
      pass_dir: first.get('orbitProperties_pass'),
      ver: CONFIG.VERSION
    });
  }

  return {s2: s2, ls: ls, s1: s1, goodKeysOf: goodKeysOf, optRow: optRow, s1Row: s1Row};
}

function runTimeseries() {
  // ROI check: JRC ever-water inside the ROI vs inside a wider zone (warning, not proof).
  var jrcMax = maxExt.gt(0);
  var jrcAreaKm2 = function(geom) {
    return ee.Number(jrcMax.multiply(ee.Image.pixelArea()).reduceRegion({
      reducer: ee.Reducer.sum(), geometry: geom, scale: 30, maxPixels: 1e9
    }).get('max_extent')).divide(1e6);
  };
  print('ROI polygon area (km2):', roi.area(1).divide(1e6));
  print('JRC max-extent water inside ROI (km2):', jrcAreaKm2(roi));
  print('JRC max-extent water inside ROI + wider zone (km2):', jrcAreaKm2(area));
  Map.addLayer(jrcMax.selfMask(), {palette: ['00ffff']}, 'JRC max extent', false, 0.5);

  var T = makeTools();
  var parts = [];
  if (CONFIG.USE_S2) {
    print('Sentinel-2 days before the cloud filter:', T.s2.aggregate_array('key').distinct().size());
    parts.push(ee.FeatureCollection(
        T.goodKeysOf(T.s2, 'G', CONFIG.SCALE).map(T.optRow(T.s2, 'S2', CONFIG.SCALE))));
  }
  if (CONFIG.USE_LANDSAT) {
    print('Landsat 8/9 days before the cloud filter:', T.ls.aggregate_array('key').distinct().size());
    parts.push(ee.FeatureCollection(
        T.goodKeysOf(T.ls, 'G', CONFIG.SCALE_LANDSAT).map(T.optRow(T.ls, 'L89', CONFIG.SCALE_LANDSAT))));
  }
  if (CONFIG.USE_S1) {
    print('Sentinel-1 day/orbit groups:', T.s1.aggregate_array('key').distinct().size());
    parts.push(ee.FeatureCollection(
        T.goodKeysOf(T.s1, 'VV', CONFIG.SCALE).map(T.s1Row)));
  }
  var rows = parts.reduce(function(a, b) { return a.merge(b); }).sort('date_ms');

  var COLS = ['day', 'date_ms', 'year', 'month', 'sensor', 'thr',
              'area_otsu', 'area_best', 'area_fixed', 'snow_km2', 'out_zone_km2',
              'valid_km2', 'roi_km2', 'zone_km2', 'valid_zone_km2', 'n_scenes',
              'orbit', 'pass_dir', 'scene_ids', 'ver'];
  var name = CONFIG.EXPORT_NAME + '_' + CONFIG.VERSION + '_' + CONFIG.START + '_' + CONFIG.END;
  Export.table.toDrive({
    collection: rows, description: name, fileNamePrefix: name,
    folder: CONFIG.EXPORT_FOLDER, fileFormat: 'CSV', selectors: COLS
  });
  if (CONFIG.SHOW_CHART) {
    print(ui.Chart.feature.groups(rows, 'date_ms', 'area_best', 'sensor')
            .setChartType('ScatterChart')
            .setOptions({title: 'Rough water area (km2), by sensor', vAxis: {title: 'km2'}, pointSize: 3}));
  }
  // Columns: area_best = recommended area. out_zone_km2 < 0.5 = trusted Sentinel-1 row.
  // valid_zone_km2 / zone_km2 = completeness (use >= 0.97).
}

// =====================================================================
// MODE = 'blocks'   ONE-TIME BLOCK SPLIT (run once, then freeze as an asset)
// =====================================================================
function runBlocks() {
  var grid = area.coveringGrid(ee.Projection(CONFIG.CRS), CONFIG.BLOCK_M)
                 .randomColumn('rnd', CONFIG.SEED);
  print('Area of the first block (km2). Expected ' + (CONFIG.BLOCK_M * CONFIG.BLOCK_M / 1e6) + ':',
        grid.first().geometry().area(1).divide(1e6));
  var gridShore = shoreZone.rename('shore').reduceRegions({
    collection: grid, reducer: ee.Reducer.max(), scale: 30
  }).map(function(f) {
    var m = ee.Number(ee.Algorithms.If(f.get('max'), f.get('max'), 0));
    return f.set({has_shore: m.toInt(), block_id: f.get('system:index')});
  });
  function split(group) {
    group = group.sort('rnd');
    var n = group.size();
    var k = n.multiply(CONFIG.TEST_SHARE).round();
    var lst = group.toList(n);
    var test  = ee.FeatureCollection(lst.slice(0, k)).map(function(f) { return f.set('role', 'test'); });
    var train = ee.FeatureCollection(lst.slice(k)).map(function(f) { return f.set('role', 'train'); });
    return test.merge(train);
  }
  var blocks = split(gridShore.filter(ee.Filter.eq('has_shore', 1)))
      .merge(split(gridShore.filter(ee.Filter.eq('has_shore', 0))));
  var role = roleImage(blocks);

  Map.addLayer(shoreZone.selfMask(), {palette: ['yellow']}, 'shore zone', false, 0.5);
  Map.addLayer(role.clip(area), {min: 0, max: 2, palette: ['00aa00', 'ff0000', 'ffaa00']},
               'role: green=train, red=test, orange=buffer', true, 0.4);
  print('Blocks total:', blocks.size());
  print('Blocks by role:', blocks.aggregate_histogram('role'));
  print('Blocks that contain shore, by role:',
        blocks.filter(ee.Filter.eq('has_shore', 1)).aggregate_histogram('role'));
  print('Shore-zone pixels by role (0=train, 1=test, 2=buffer):',
        role.updateMask(shoreZone).reduceRegion({
          reducer: ee.Reducer.frequencyHistogram(), geometry: area,
          scale: 30, maxPixels: 1e9, tileScale: 4
        }).get('role'));
  if (CONFIG.ASSET_ID) {
    Export.table.toAsset({
      collection: blocks.select(['block_id', 'role', 'has_shore', 'rnd']),
      description: 'sarsang_blocks_' + CONFIG.VERSION, assetId: CONFIG.ASSET_ID
    });
  }
}

// =====================================================================
// MODE = 'build'   THE DATASET  -> ONE CSV
// =====================================================================
function runBuild() {
  var role = roleImage(ee.FeatureCollection(CONFIG.BLOCKS_ASSET));
  var s1all = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(roi)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .map(function(i) {   // key = day_orbit_pass
        var key = i.date().format('YYYY-MM-dd').cat('_')
            .cat(ee.Number(i.get('relativeOrbitNumber_start')).format('%d')).cat('_')
            .cat(ee.String(i.get('orbitProperties_pass')));
        return i.set('key', key);
      });
  var PROJ10 = ee.Projection(CONFIG.CRS).atScale(10);   // 10 m grid, pixel centres
  var S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
  var FEATURE_COLS = S2_BANDS.concat(['VV', 'VH', 'mndwi', 'vv_s', 'elev', 'slope']);   // no jrc_occ
  var elev = dem.select('elevation');

  function buildDate(cfg, k) {
    var dateStr = cfg.d, s1key = cfg.s1;
    var t0 = ee.Date(dateStr);
    var s2c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi).filterDate(t0, t0.advance(1, 'day'))
        .linkCollection(csp, ['cs_cdf'])
        .map(function(i) { return i.updateMask(i.select('cs_cdf').gte(CONFIG.CS_MIN)); });
    var s2 = s2c.mosaic();
    var d1 = ee.Date(s1key.split('_')[0]);
    var s1c = s1all.filterDate(d1, d1.advance(1, 'day')).filter(ee.Filter.eq('key', s1key));
    var s1m = s1c.select(['VV', 'VH']).mosaic();
    print(dateStr + ' | S2 scenes:', s2c.size(), '| S1 scenes (' + s1key + '):', s1c.size());

    var g = s2.select('B3').divide(10000).max(0);
    var sw = s2.select('B11').divide(10000).max(0);
    var mndwi = g.subtract(sw).divide(g.add(sw).max(0.0001)).rename('mndwi');
    var snowLike = s2.select('SCL').eq(11).or(mndwi.gt(0.4).and(g.gt(0.25))).unmask(0);
    var vvS = s1m.select('VV').focal_median(30, 'circle', 'meters').rename('vv_s');   // labels only

    var valid = g.mask().unmask(0).and(sw.mask().unmask(0)).and(vvS.mask().unmask(0));
    var water = mndwi.gt(CONFIG.MNDWI_WATER).and(vvS.lt(CONFIG.S1_WATER_DB)).unmask(0);
    var land  = mndwi.lt(CONFIG.MNDWI_LAND).and(vvS.gt(CONFIG.S1_LAND_DB)).unmask(0);

    var inShore = shoreZone;
    var okShore = valid.and(inShore);
    var agreeW  = okShore.and(water).and(snowLike.not());
    var agreeL  = okShore.and(land).and(snowLike.not());
    var hard    = okShore.and(agreeW.not()).and(agreeL.not());
    var other   = valid.and(inShore.not());

    // 1 agree water | 0 agree land | -1 hard | 2 outside shore zone | 9 invalid
    var stratum = ee.Image(9)
        .where(agreeW, 1).where(agreeL, 0).where(hard, -1).where(other, 2)
        .toInt().rename('stratum');

    // training classes with the conflict filters
    var conflictW = mndwi.lt(CONFIG.MNDWI_LAND).and(vvS.gt(CONFIG.S1_WATER_DB)).unmask(0);
    var conflictL = water.and(snowLike.not());
    var shadowW   = elev.gt(CONFIG.WATER_REF_M + CONFIG.SHADOW_ABOVE_M).unmask(0);
    // 1 weak water | 0 weak land | 3 stable water (T1) | 4 far land (T1)
    var tstr = ee.Image(99)
        .where(stratum.eq(1).and(shadowW.not()), 1)
        .where(stratum.eq(0), 0)
        .where(valid.and(stableWater).and(conflictW.not()), 3)
        .where(valid.and(landFar).and(conflictL.not()), 4);
    tstr = tstr.updateMask(tstr.neq(99)).toInt().rename('tstratum');

    var featImg = ee.Image.cat([
        s2.select(S2_BANDS).divide(10000),
        s1m.select(['VV', 'VH']),
        mndwi, vvS,
        elev.rename('elev'),
        ee.Terrain.slope(dem).rename('slope'),
        inShore.rename('in_shore')
    ]).toFloat();

    function withFeatures(pts, prop) {
      return featImg.sampleRegions({
        collection: pts, properties: [prop], projection: PROJ10, tileScale: 4, geometries: true
      }).map(function(f) {
        var c = ee.List(f.geometry().coordinates());
        return f.set({lon: c.get(0), lat: c.get(1), date: dateStr, k: k});
      });
    }

    // ---- pixel counts of every stratum in the TEST frame (the weights) ----
    var h = ee.Dictionary(stratum.updateMask(role.eq(1)).reduceRegion({
      reducer: ee.Reducer.frequencyHistogram(), geometry: area, crs: CONFIG.CRS,
      scale: 10, maxPixels: 1e10, tileScale: 8
    }).get('stratum'));
    var framePx = ee.Number(h.get('1', 0)).add(ee.Number(h.get('0', 0)))
        .add(ee.Number(h.get('-1', 0))).add(ee.Number(h.get('2', 0))).add(ee.Number(h.get('9', 0)));

    // ---- label rows (test blocks only, NOT filtered) ----
    var labelBase;
    if (CONFIG.LABEL_POINTS_FROM_ASSET) {
      labelBase = ee.FeatureCollection(CONFIG.LABEL_ASSET).filter(ee.Filter.eq('date', dateStr));
    } else {
      var N = CONFIG.N_LABEL;
      var labelRaw = stratum.updateMask(role.eq(1)).stratifiedSample({
        numPoints: 0, classBand: 'stratum', region: area, projection: PROJ10,
        seed: CONFIG.SEED + k, classValues: [1, 0, -1, 2],
        classPoints: [N.aw, N.al, N.hard, N.other], geometries: true, tileScale: 4
      });
      labelBase = withFeatures(labelRaw, 'stratum');
    }
    var labelPts = labelBase.map(function(f) {
      var key = ee.Number(f.get('stratum')).format('%d');
      return f.set({set: 'label', stratum_px: ee.Number(h.get(key, 0)), frame_px: framePx,
                    dk: ee.String(dateStr).cat('|').cat(key)});
    });

    // ---- train rows (train blocks only, filtered) ----
    var M = CONFIG.N_TRAIN;
    var trainRaw = tstr.updateMask(role.eq(0)).stratifiedSample({
      numPoints: 0, classBand: 'tstratum', region: area, projection: PROJ10,
      seed: CONFIG.SEED + 100 + k, classValues: [1, 0, 3, 4],
      classPoints: [M.t2w, M.t2l, M.t1w, M.t1l], geometries: true, tileScale: 4
    });
    var trainPts = withFeatures(trainRaw, 'tstratum').map(function(f) {
      var ts = ee.Number(f.get('tstratum'));
      var tier = ee.String(ee.Algorithms.If(ts.eq(1), 'T2_water',
                 ee.Algorithms.If(ts.eq(0), 'T2_land',
                 ee.Algorithms.If(ts.eq(3), 'T1_water', 'T1_land'))));
      var isWater = ee.Number(ee.Algorithms.If(ts.eq(1).or(ts.eq(3)), 1, 0));
      return f.set({set: 'train', tier: tier, label_weak: isWater,
                    dk: ee.String(dateStr).cat('|').cat(tier)});
    });

    return {stratum: stratum, labelPts: labelPts, trainPts: trainPts};
  }

  var idxs = CONFIG.RUN_INDICES;
  if (idxs === null || idxs === undefined) {
    idxs = CONFIG.DEV_DATES.map(function(x, i) { return i; });
  }
  print('Dates in this run:', idxs.map(function(i) { return CONFIG.DEV_DATES[i].d; }));
  print('Hold-out dates (NOT used here):', CONFIG.HOLDOUT_DATES);

  var labelAll = null, trainAll = null, preview = null;
  idxs.forEach(function(i) {
    var r = buildDate(CONFIG.DEV_DATES[i], i);
    labelAll = labelAll ? labelAll.merge(r.labelPts) : r.labelPts;
    trainAll = trainAll ? trainAll.merge(r.trainPts) : r.trainPts;
    if (i === CONFIG.PREVIEW_INDEX || preview === null) { preview = {i: i, r: r}; }
  });

  print('TRAIN rows per date|tier (target per date: ' + JSON.stringify(CONFIG.N_TRAIN) + '):',
        trainAll.aggregate_histogram('dk'));
  print('LABEL rows per date|stratum (target per date: ' + JSON.stringify(CONFIG.N_LABEL) + '):',
        labelAll.aggregate_histogram('dk'));

  Map.addLayer(roleImage(ee.FeatureCollection(CONFIG.BLOCKS_ASSET)).clip(area),
               {min: 0, max: 2, palette: ['00aa00', 'ff0000', 'ffaa00']},
               'role: green=train, red=test, orange=buffer', false, 0.3);
  Map.addLayer(preview.r.stratum.updateMask(preview.r.stratum.neq(9)).clip(area),
               {min: -1, max: 2, palette: ['ff00ff', '00aa00', '0000ff', 'ffffff']},
               'strata (' + CONFIG.DEV_DATES[preview.i].d + '): magenta=hard, green=land, blue=water, white=outside zone',
               true, 0.6);
  Map.addLayer(preview.r.labelPts.style({color: 'ffff00', pointSize: 3}), {}, 'label points (preview date)');
  Map.addLayer(preview.r.trainPts.style({color: '00ffff', pointSize: 2}), {}, 'train points (preview date)', false);

  // ---- THE ONE CSV ----
  var ALL_COLS = ['set', 'date', 'k', 'lon', 'lat', 'tier', 'label_weak', 'stratum', 'in_shore',
                  'stratum_px', 'frame_px'].concat(FEATURE_COLS);
  Export.table.toDrive({
    collection: trainAll.merge(labelAll),
    description: 'sarsang_dataset_' + CONFIG.VERSION,
    fileNamePrefix: 'sarsang_dataset_' + CONFIG.VERSION,
    folder: CONFIG.EXPORT_FOLDER, fileFormat: 'CSV', selectors: ALL_COLS
  });

  if (CONFIG.EXPORT_LABEL_ASSET) {   // fresh start only (the labeling tool reads this asset)
    var LABEL_COLS = ['date', 'k', 'lon', 'lat', 'stratum', 'in_shore'].concat(FEATURE_COLS);
    Export.table.toAsset({
      collection: labelAll.select(LABEL_COLS),
      description: 'sarsang_label_pts_' + CONFIG.VERSION,
      assetId: CONFIG.ASSET_FOLDER + '/sarsang_label_pts_' + CONFIG.VERSION
    });
  }
  // Expected rows in the CSV: 8 dates x (380 train + 90 label) = 3760.
  // Do not change thresholds or N_* after labeling started: the labels belong to this design.
}

// =====================================================================
// MODE = 'label'   BLIND MANUAL LABELING (one date per session)
// =====================================================================
// PROTOCOL (keep it the same for all dates). Look at the pixel inside the magenta square.
//   Water : open water, also shallow water you can see through.
//   Land  : soil, sand, rock, plants, WET MUD or wet sand without standing water.
//   Unsure: cloud, cloud shadow, snow, or a half water / half land pixel. Do not guess.
function runLabel() {
  var FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'VV', 'VH', 'mndwi', 'vv_s', 'elev', 'slope'];
  var DATE = CONFIG.DEV_DATES[CONFIG.DATE_INDEX].d;
  var S1KEY = CONFIG.DEV_DATES[CONFIG.DATE_INDEX].s1;

  var fc = ee.FeatureCollection(CONFIG.LABEL_ASSET)
      .filter(ee.Filter.eq('date', DATE))
      .randomColumn('ord', CONFIG.ORDER_SEED)
      .sort('ord');
  var allPts = fc.getInfo().features;
  var nUse = Math.ceil(allPts.length * CONFIG.SUBSET_SHARE);
  var pts = allPts.slice(0, nUse);
  print('Date ' + DATE + ' | points in asset: ' + allPts.length + ' | used now: ' + pts.length +
        ' | annotator ' + CONFIG.ANNOTATOR + '. Expected about 90 points per date.');

  // Images of this date, NO cloud mask (you must see clouds and shadows)
  var t0 = ee.Date(DATE);
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi).filterDate(t0, t0.advance(1, 'day')).mosaic();
  var d1 = ee.Date(S1KEY.split('_')[0]);
  var s1col = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(roi).filterDate(d1, d1.advance(1, 'day'))
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.eq('relativeOrbitNumber_start', parseInt(S1KEY.split('_')[1], 10)))
      .filter(ee.Filter.eq('orbitProperties_pass', S1KEY.split('_')[2]));
  var s1vv = s1col.select('VV').mosaic();

  var mapRGB = ui.Map(), mapSWIR = ui.Map(), mapS1 = ui.Map();
  var maps = [mapRGB, mapSWIR, mapS1];
  maps.forEach(function(m) {
    m.setControlVisibility({all: false, zoomControl: true, mapTypeControl: false});
    m.setOptions('SATELLITE');
    m.style().set('cursor', 'crosshair');
  });
  mapRGB.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'S2 true colour ' + DATE);
  mapSWIR.addLayer(s2, {bands: ['B12', 'B8', 'B4'], min: 0, max: 4000}, 'S2 SWIR-NIR-red ' + DATE);
  mapS1.addLayer(s1vv, {min: -25, max: 0}, 'S1 VV ' + S1KEY);
  ui.Map.Linker(maps);
  var grid = ui.Panel(
      maps.map(function(m) { m.style().set({width: '33%', height: '100%'}); return m; }),
      ui.Panel.Layout.Flow('horizontal'), {stretch: 'both'});
  ui.root.widgets().reset([grid]);

  var results = new Array(pts.length);
  var idx = CONFIG.START_INDEX;
  var MARK = 1;
  var progress = ui.Label('');
  var savedLabel = ui.Label('');

  function setMarker(m, layer) {
    var L = m.layers();
    if (L.length() > MARK) { L.set(MARK, layer); } else { L.add(layer); }
  }
  function pixelSquare(lon, lat) {   // hollow 10 m square on the pixel (point = pixel centre)
    var dLat = 5 / 110540;
    var dLon = 5 / (111320 * Math.cos(lat * Math.PI / 180));
    var rect = ee.Geometry.Rectangle([lon - dLon, lat - dLat, lon + dLon, lat + dLat]);
    return ee.FeatureCollection([ee.Feature(rect)])
        .style({color: 'ff00ff', fillColor: '00000000', width: 2});
  }
  function refresh() {
    if (idx >= pts.length) { progress.setValue('All done. Press "Print CSV".'); return; }
    var c = pts[idx].geometry.coordinates;
    var sq = pixelSquare(c[0], c[1]);
    maps.forEach(function(m) { setMarker(m, ui.Map.Layer(sq, {}, 'pixel to label')); });
    mapRGB.setCenter(c[0], c[1], CONFIG.ZOOM);
    progress.setValue('Point ' + (idx + 1) + ' / ' + pts.length + '   date ' + DATE +
                      '   annotator ' + CONFIG.ANNOTATOR);
  }
  function exportCsv() {
    var BASE = ['annotator', 'date', 'order', 'lon', 'lat', 'label', 'stratum', 'in_shore'];
    var rows = [BASE.concat(FEATURE_COLS).join(',')];
    var n = 0;
    results.forEach(function(r) {
      if (!r) { return; }
      var vals = [r.annotator, r.date, r.order, r.lon, r.lat, r.label, r.stratum, r.in_shore];
      FEATURE_COLS.forEach(function(k) {
        var v = r.props[k];
        vals.push((v === undefined || v === null) ? '' : v);
      });
      rows.push(vals.join(','));
      n++;
    });
    print('--- CSV (' + n + ' labeled of ' + pts.length + ') copy the LAST block into val_' + DATE +
          '_' + CONFIG.ANNOTATOR + '.csv ---');
    print(rows.join('\n'));
  }
  function answer(v) {
    if (idx >= pts.length) { return; }
    var p = pts[idx];
    results[idx] = {
      annotator: CONFIG.ANNOTATOR, date: DATE, order: idx,
      lon: p.geometry.coordinates[0], lat: p.geometry.coordinates[1],
      label: v, stratum: p.properties.stratum, in_shore: p.properties.in_shore,
      props: p.properties
    };
    idx++;
    var done = results.filter(function(r) { return r; }).length;
    savedLabel.setValue('Labeled in this session: ' + done);
    if (done > 0 && done % CONFIG.AUTOPRINT_EVERY === 0) { exportCsv(); }
    refresh();
  }

  var panel = ui.Panel({style: {position: 'top-right', width: '270px'}});
  panel.add(ui.Label('What is the pixel inside the magenta square?', {fontWeight: 'bold'}));
  panel.add(progress);
  panel.add(savedLabel);
  panel.add(ui.Button({label: 'Water (1)', onClick: function() { answer(1); }}));
  panel.add(ui.Button({label: 'Land (0)', onClick: function() { answer(0); }}));
  panel.add(ui.Button({label: 'Unsure (-1)', onClick: function() { answer(-1); }}));
  panel.add(ui.Button({label: 'Back', onClick: function() {
    if (idx > CONFIG.START_INDEX) { idx--; results[idx] = null; refresh(); }
  }}));
  panel.add(ui.Button({label: 'Print CSV (Console)', onClick: exportCsv}));
  panel.add(ui.Label(
      'Water: open water, also shallow water. Land: soil, sand, rock, plants, WET MUD. ' +
      'Unsure: cloud, shadow, snow, or a half water / half land pixel. Do not guess.',
      {fontSize: '11px'}));
  mapS1.add(panel);
  refresh();
}

// ---------- RUN ----------
Map.centerObject(roi, 12);
Map.setOptions('SATELLITE');
Map.addLayer(roi, {color: 'yellow'}, 'ROI', true, 0.3);
if      (MODE === 'timeseries') { runTimeseries(); }
else if (MODE === 'blocks')     { runBlocks(); }
else if (MODE === 'build')      { runBuild(); }
else if (MODE === 'label')      { runLabel(); }
else { print('MODE must be "timeseries", "blocks", "build" or "label". Current value:', MODE); }