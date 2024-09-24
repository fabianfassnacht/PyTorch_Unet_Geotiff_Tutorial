# load necessary package
require(terra)

# get file names and path to all predicted tiles
fils <- list.files("D:/0_Tutorial/4_output/3_continuos_predictions", pattern=".tif$", full.names = T)
# create empty list
fils2 <- list()
# loop through all files and load them as rasterfile to list
for (i in 1:length(fils)){
  
  img <- rast(fils[i])
  fils2[[i]] <- img
  
}

# add a function term to the list - this will decide which which
# function overlapping pixels will be summarized in the mosaic function called
# below
fils2$fun <- "median"

# call mosaic function
img_end <- do.call(mosaic, fils2)
# see result
plot(img_end) 
# save resulting mosaic to file
setwd("D:/0_Tutorial/4_output/4_prediction_maps")
writeRaster(img_end, filename = "predictions_256_tiles_30epochs_median.tif")


# optionally you can apply a threshold to create a binary mask
# this can be for example done after manually checking the resulting
# prediction map in QGIS and identifying a meaningful threshold
img_end_th1 <- img_end > -1300
# save binary mask to file
writeRaster(img_end_th1, filename = "predictions_20epochs_gt_m1300.tif")


