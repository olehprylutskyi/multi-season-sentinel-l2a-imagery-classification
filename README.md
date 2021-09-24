# Multi-season Sentinel-2 level 2a image stack classification using Random Forest in Google Earth Engine
Script for Google Earth Engine to obtain a multi-season image stack and classify it (with user-defined ground truth) using Random Forest machine learning approach
 
## Requirements
For running, the script requires user-defined Area Of Interest (spatial polygon) and ground truth data (spatial polygons with numerical classes). Both files should be loaded into GEE environment as separate assets.
 
For correct displaying of classified imagery one should manually edit the palette, using  desired class color codes and ajust the range of coloring according to the number of classes.
 
## Outputs
1. Classified satellite imagery for user-defined area of interest (exportable to Google Drive as .tiff raster file)
2. Variable importance chart and values
3. Overall accuracy metric
4. Cohen's Kappa metric
5. Confusion matrix (explortable to Google Drive as .CSV file)
 
## Example of results
Southern Buh river valley, Ukraine (47.96 N, 31.03 E).
 
![Map of 26 EUNIS habitat types](https://github.com/olehprylutskyi/multi-season-sentinel-l2a-imagery-classification/blob/main/exampleresults.png)

## Contributors
[Anton Biatov](https://github.com/abiatov) - coding

[Dariia Shyriaieva](darshyr@gmail.com) - idea and ground truth
 
 