require(terra)
require(caret)

setwd("D:/0_Tutorial/6_model_comparison/evaluation")

# Load your binary raster layers
raster1 <- rast("mask_4_2_3.tif")
raster2 <- rast("pred_4_2_3.tif")

# get value range of prediction raster and restrict to 1 and 99% percentile
percentiles <- quantile(values(raster2), probs = c(0.01, 0.99), na.rm = TRUE)
# create 1000 thresholds to test
th <- seq( percentiles[1],percentiles[2],(as.numeric(percentiles[2]-percentiles[1])/999)) 

# create empty list to store results
IoU_res <- list()

# iterate through thresholds
for (i in 1:1000){
  
  # apply current treshold and create binary raster with value 0 or 1
  raster3 <- raster2 > th[i]
  raster3 <- as.numeric(raster3)

  # Calculate the intersection (logical AND)
  intersection <- raster1 & raster3
  
  # Calculate the union (logical OR)
  union <- raster1 | raster3
  
  # calculate IoU helpers
  intersection_area <- global(intersection, fun = sum, na.rm = TRUE)[,1]
  union_area <- global(union, fun = sum, na.rm = TRUE)[,1]
  
  # Calculate IoU
  iou <- intersection_area / union_area
  
  # calaculate traditional pixel-based confusion matrix
  vec1 <- as.factor(as.vector(raster1))
  vec2 <- as.factor(as.vector(raster3))
  cfm <- confusionMatrix(vec1, vec2)
  
  # extract kappy value
  cfm$overall[2]
  
  # Save IoU and kappa value to list
  IoU_res[[i]] <- c(iou, cfm$overall[2])
  
  # print current iteration (i of 1000 tresholds)
  print(i)
  

}

# prepare data
res <- do.call(rbind, IoU_res)
# plot kappa values
plot(seq(1,1000,1), res[,2])
# get highest kappa value
best <- which(res[,2] == max(res[,2]))
res[best]
th[best]

# apply threshold of highest kappa value
raster_best <- raster2 > th[best]
# check plots after applying best threshold according to kappa
plot(raster_best)
plot(raster1)
