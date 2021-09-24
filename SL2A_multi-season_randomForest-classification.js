
Map.setOptions("HYBRID"); // set default map background

//  SET Area Of Interest
//  Area of interest within we are going to classify habitat types

var AOI = ee.FeatureCollection('users/<your-GEE-account>/<asset-name-for-AOI>');

var name_area = '<verbatin-short-area-name-w/o-spaces>'; // area name, which will be used in exported file naming
var criteria = 'eunis2020'; // habitat classification system or other human-readable classification system name

Map.centerObject(AOI, 12); // center map preview


//  SET PRIORS

var year = 2019; // which year will be used for receiving imageries

// Set time periods in which median satellite imageries will be generated. 

/*
Here we operated 8 month-long periods, from March, 15, to November, 15.
Script seems to be working only with up to 5 time period (retrieves an error of out of memory if more)
Days of year must be defined as Julian day numbers. I used following link:

https://people.biology.ucsd.edu/patrick/julian_cal.html

Vegetative period

105-135 - (15 Apr - 15 May) 
135-166 - (15 May - 15 Jun)
166-196 - (15 Jun - 15 Jul)
196-227 - (15 Jul - 15 Aug)
227-258 - (15 Aug - 15 Sep)
258-288 - (15 Sep - 15 Oct)

leaf-less preriod

074-105 - (15 Mar - 15 Apr)
288-319 - (15 Oct - 15 Nov)
*/

var STARTDAY_1 = 74; //julian number of start day of the year
var ENDDAY_1 = 105; //julian number of end day of the year

var STARTDAY_2 = 105; 
var ENDDAY_2 = 135; 

var STARTDAY_3 = 135; 
var ENDDAY_3 = 166; 

var STARTDAY_4 = 166; 
var ENDDAY_4 = 196; 

var STARTDAY_5 = 196; 
var ENDDAY_5 = 227; 

var STARTDAY_6 = 227; 
var ENDDAY_6 = 258; 

var STARTDAY_7 = 258; 
var ENDDAY_7 = 288;

var STARTDAY_8 = 288; 
var ENDDAY_8 = 319; 


var cloud_treshold = 10; // max per cent of sky covered by clouds

//  CLOUD MASK

/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 2 << 10;
  var cirrusBitMask = 2 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask);
//  return image.updateMask(mask).divide(10000); // если делить на 10000 то результирующий растр будет в долях еденицы.
  
}

// Receiving median satellite imageries for defined timespans 


var pre_med_1 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_1, ENDDAY_1))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8_1', 'B8A_1', 'B11_1', 'B12_1');
  print('pre_med_1: ', pre_med_1);


var pre_med_2 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_2, ENDDAY_2))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_2', 'B3_2', 'B4_2', 'B5_2', 'B6_2', 'B7_2', 'B8_2', 'B8A_2', 'B11_2', 'B12_2');
  print('pre_med_2: ', pre_med_2);
    
    
var pre_med_3 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_3, ENDDAY_3))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_3', 'B3_3', 'B4_3', 'B5_3', 'B6_3', 'B7_3', 'B8_3', 'B8A_3', 'B11_3', 'B12_3');
  print('pre_med_3: ', pre_med_3);


var pre_med_4 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_4, ENDDAY_4))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_4', 'B3_4', 'B4_4', 'B5_4', 'B6_4', 'B7_4', 'B8_4', 'B8A_4', 'B11_4', 'B12_4');
  print('pre_med_4: ', pre_med_4);

var pre_med_5 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_5, ENDDAY_5))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_5', 'B3_5', 'B4_5', 'B5_5', 'B6_5', 'B7_5', 'B8_5', 'B8A_5', 'B11_5', 'B12_5');
  print('pre_med_5: ', pre_med_5);
  
  
var pre_med_6 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_6, ENDDAY_6))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_6', 'B3_6', 'B4_6', 'B5_6', 'B6_6', 'B7_6', 'B8_6', 'B8A_6', 'B11_6', 'B12_6');
  print('pre_med_6: ', pre_med_6);

var pre_med_7 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_7, ENDDAY_7))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_7', 'B3_7', 'B4_7', 'B5_7', 'B6_7', 'B7_7', 'B8_7', 'B8A_7', 'B11_7', 'B12_7');
  print('pre_med_7: ', pre_med_7);

var pre_med_8 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.dayOfYear(STARTDAY_8, ENDDAY_8))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_treshold))
                  .map(maskS2clouds)
                  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                  .median()
                  .rename('B2_8', 'B3_8', 'B4_8', 'B5_8', 'B6_8', 'B7_8', 'B8_8', 'B8A_8', 'B11_8', 'B12_8');
  print('pre_med_8: ', pre_med_8);


// Merging satellite bands from different imageries into single multi-band stack

var med = pre_med_1.addBands(pre_med_2).addBands(pre_med_3).addBands(pre_med_4).addBands(pre_med_5)
.addBands(pre_med_6).addBands(pre_med_7).addBands(pre_med_8)
    .clip(AOI)
    print('med: ', med);


// Set visual previev of obtained imagery

var rgbVis_natural_color = {
  min: 0,
  max: 1500,
  bands: ['B4_3', 'B3_3', 'B2_3'],
};


//  Add several layers to the map


Map.addLayer(med/*.clip(AOI)*/, rgbVis_natural_color, 'natural_color');

// Add vector boundary of AOI

var empty = ee.Image().byte();
var outline = empty.paint({
  featureCollection: AOI,
  color: 1,
  width: 2
});

Map.addLayer(outline, {palette: 'FF0000'}, 'AOI');


// SUPERVISED CLASSIFICATION

// Load vector layer from assets (must be polygons and contain ClassInt variable (integer), with numerical class ID)

var train_fc = ee.FeatureCollection('users/<your-GEE-account>/<asset-name-for-ground-truth-polygons>');

print('Training areas', train_fc);

// Select the bands for training
/*
If you assign "var bands" as following, you will obtain 80-band raster stack. GEE retrieve 
an error with all those bands. With my AOI and ground truth, I am able to use up to 
50-band image, so I manually assigned desired bands to the var bands 
*/

/* All set of available bands

var bands = ['B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8_1', 'B8A_1', 'B11_1', 'B12_1', 
'B2_2', 'B3_2', 'B4_2', 'B5_2', 'B6_2', 'B7_2', 'B8_2', 'B8A_2', 'B11_2', 'B12_2', 
'B2_3', 'B3_3', 'B4_3', 'B5_3', 'B6_3', 'B7_3', 'B8_3', 'B8A_3', 'B11_3', 'B12_3', 
'B2_4', 'B3_4', 'B4_4', 'B5_4', 'B6_4', 'B7_4', 'B8_4', 'B8A_4', 'B11_4', 'B12_4', 
'B2_5', 'B3_5', 'B4_5', 'B5_5', 'B6_5', 'B7_5', 'B8_5', 'B8A_5', 'B11_5', 'B12_5', 
'B2_6', 'B3_6', 'B4_6', 'B5_6', 'B6_6', 'B7_6', 'B8_6', 'B8A_6', 'B11_6', 'B12_6', 
'B2_7', 'B3_7', 'B4_7', 'B5_7', 'B6_7', 'B7_7', 'B8_7', 'B8A_7', 'B11_7', 'B12_7', 
'B2_8', 'B3_8', 'B4_8', 'B5_8', 'B6_8', 'B7_8', 'B8_8', 'B8A_8', 'B11_8', 'B12_8'];
*/

// Selected bands

var bands = ['B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8_1', 'B8A_1', 'B11_1', 'B12_1',
'B2_3', 'B3_3', 'B4_3', 'B5_3', 'B6_3', 'B7_3', 'B8_3', 'B8A_3', 'B11_3', 'B12_3',
'B2_5', 'B3_5', 'B4_5', 'B5_5', 'B6_5', 'B7_5', 'B8_5', 'B8A_5', 'B11_5', 'B12_5',
'B2_7', 'B3_7', 'B4_7', 'B5_7', 'B6_7', 'B7_7', 'B8_7', 'B8A_7', 'B11_7', 'B12_7', 
'B2_8', 'B3_8', 'B4_8', 'B5_8', 'B6_8', 'B7_8', 'B8_8', 'B8A_8', 'B11_8', 'B12_8'];

// Sample the input imagery to get a FeatureCollection of training data.

var sampled = med.select(bands).sampleRegions({
  collection: train_fc,
  properties: ['ClassInt'],
// geometries: true,
  scale: 10,
});


/*
The first step is to partition the set of known values into training and testing sets.
Reusing the classification training set, add a column of random numbers used to 
partition the known data where about 60% of the data will be used for training and 
40% for testing:
*/

var trainingTesting = sampled.randomColumn({
  seed: 1 // set seed for reproducible partitioning
});

var training = trainingTesting
.filter(ee.Filter.lessThan('random', 0.6));
var testing = trainingTesting
.filter(ee.Filter.greaterThanOrEquals('random', 0.6));

// Print an amount of pixels in training and testing sapmples
print('All sampled pixels:', sampled.size())
print('Training data', training.size());
print('Testing areas', testing.size());

/*
// Export training samples to Google Drive
Export.table.toDrive({
  collection: training,
  folder: 'GEE data',
  description:'Training_S2_RF_'+criteria+'_'+year+'_'+STARTDAY+'_'+ENDDAY,
  fileFormat: 'CSV'
});
*/


//  RANDOM FOREST

// Make a Random Forest classifier with fixed number of trees
var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 30,
});


//  Train the RF classifier.
var trained = classifier.train({
  features: training,
  classProperty: 'ClassInt',
  inputProperties: bands
});


// Classify the input imagery.
var classified = med.select(bands).classify(trained);

print('classified: ', trained);

/*
// Compute mode filter of classified, if ypu need it.
var filtred = classified.reduceNeighborhood({
  reducer: ee.Reducer.mode(),
//  kernel: ee.Kernel.circle(3,  'pixels' ),
  kernel: ee.Kernel.rectangle(20, 20,  'meters' ),
});
*/


// Variable importance
 
var dict = trained.explain();

print('Explain:',dict);
 
var variable_importance = ee.Feature(null, ee.Dictionary(dict).get('importance'));
 
var chart =
ui.Chart.feature.byProperty(variable_importance)
.setChartType('ColumnChart')
.setOptions({
title: 'Random Forest Variable Importance',
legend: {position: 'none'},
hAxis: {title: 'Bands'},
vAxis: {title: 'Importance'}
});
 
print(chart);


// Accuracy assessment


/*
// Set and export the confusion matrix to CSV file

var trainAccuracy = trained.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy);
print('Training overall accuracy: ', trainAccuracy.accuracy());

var exportAccuracy = ee.Feature(null, {matrix: trainAccuracy.array()})

Export.table.toDrive({
  collection: ee.FeatureCollection(exportAccuracy), 
  description: 'Confision_matrix_'+criteria+'_'+year+'_'+STARTDAY+'_'+ENDDAY, 
  folder: 'GEE data',
  fileFormat: 'CSV',
});
*/


// Classify the testing set and get a confusion matrix. Note that the classifier 
// automatically adds a property called 'classification', which is compared to the 
// 'class' property added when you imported your polygons:

var confusionMatrix = ee.ConfusionMatrix(testing.classify(trained)
    .errorMatrix({
      actual: 'ClassInt', 
      predicted: 'classification',
    }));
    
// Print the confusion matrix and expand the object to inspect the matrix.
// The entries represent number of pixels.  Items on the diagonal represent correct 
// classification.  Items off the diagonal are misclassifications, where the class in 
// row i is classified as column j.  It's also possible to get basic descriptive 
// statistics from the confusion matrix.  For example:


print('Confusion matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa: ', confusionMatrix.kappa());



// Set and export the accuracy parameters from the confusion matrix to CSV files
// Confusion matrix
Export.table.toDrive({
  collection: ee.FeatureCollection([
      ee.Feature(null, {matrix: confusionMatrix.array()})
  ]), 
  description: 'Confision_matrix_'+criteria+'_'+year, 
  folder: 'GEE data',
  fileFormat: 'CSV',
});
// Kappa
Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(null, {matrix: confusionMatrix.kappa()})
  ]), 
  description: 'Kappa_'+criteria+'_'+year, 
  folder: 'GEE data',
  fileFormat: 'CSV',
});
// Overall Accuracy
Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(null, {matrix: confusionMatrix.accuracy()})
  ]), 
  description: 'Accuracy_'+criteria+'_'+year, 
  folder: 'GEE data',
  fileFormat: 'CSV',
});





//+++++++++++++++++++COLORIZING+++++++++++++++++++++++++++

// Define a palette for the given habitat types as html color code (without #)
// for each class.

var palette_eunis_2020 = [
  '4AA4C1', // C2.2 (1)
  '357A9A', // C2.3 (2)
  'D1D1D1', // J1 (3)  
  '989384', // J3.2 (4) 
  '606060', // J4.2 (5)
  '7E88A5', // J5 (6)
  '2B8978', // Q51 (7)  
  '35A893', // Q53 (8)
  'FFC055', // R12 (9) 
  'C1D757', // R1A (10)
  'DFC229', // R1B (11) 
  'A8E787', // R21 (12) 
  '98F4B6', // R36 (13)
  'D37568', // S35 (14)
  'E67C5B', // S36 (15)  
  'A85D53', // S91 (16)
  '315969', // T11 (17)
  '066951', // T13 (18)  
  '769730', // T19 (19)
  '548B42', // T1E (20) 
  '8A8B53', // T1H (21)
  'FFA600', // U33 (22)
  'FFE0CE', // V11 (23)  
  'F0828B', // V34 (24) 
  'DE425B', // V38 (25)   
  'DA9147', // X18 (26)
];

// Display the classification result and the input image.
Map.addLayer(classified, {min: 1, max: 26, palette: palette_eunis_2020}, 'Land Use Classification');
// Map.addLayer(filtred, {min: 1, max: 26, palette: palette_eunis_2020}, 'FILTRED',false);


// Export classificatin results to Google Drive

 Export.image.toDrive({
  image: classified,
  description: name_area+'_classified_by_'+criteria+'_'+year,
  folder: 'GEE data',
  scale: 10,
  region: AOI,
  crs: 'EPSG:4326',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

/* Filtered image
 Export.image.toDrive({
  image: filtred,
  description: name_area+'_classified_by_'+criteria+'_'+'filtered_'+year+'_'+STARTDAY+'_'+ENDDAY,
  folder: 'GEE data',
  scale: 10,
  region: AOI,
  crs: 'EPSG:4326',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});
*/

// Median raster of Sentinel L2A for the given Area of Interest and for date range

  Export.image.toDrive({
  image: med.clip(AOI),
  description: name_area+'_S2_median_'+year,
  folder: 'GEE data',
  scale: 10,
  region: AOI,
  crs: 'EPSG:4326',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
  });
