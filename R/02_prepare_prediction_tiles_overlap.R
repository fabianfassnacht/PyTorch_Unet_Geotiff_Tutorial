rm(list = ls())
# Set working data
setwd("D:/5_pytorch/pre_processing/1_prediction_data")
# load necessaries packages
library(terra)
library(sf)

# import image
parakou <- rast("pansharp_parakou_wv3_prediction_area.tif")
plot(parakou)

## import tree shapefile
res_ras <- res(parakou)[1]
	
# get extent of the prediction area
studarea <- ext(parakou)
# define tile size
tilesize = 128*res_ras
	
# define steps in a way that tiles will overlap (the divide by 9 is the key here)
xsteps <- seq(studarea[1], studarea[2], (tilesize/9))
ysteps <- seq(studarea[3], studarea[4], (tilesize/9))

# loop through the steps and create tiles
for (i1 in 1:(length(xsteps)-9)){
  for (i2 in 1:(length(ysteps)-9)){
    
    # get extent for current tile (be aware that we have to add 9 to
    # maintain the same pixel size as for the input data)
    clipext <- ext(xsteps[i1], xsteps[i1+9], ysteps[i2], ysteps[i2+9])
    
    # crop image to current tile
    img <- crop(parakou, clipext)
    # rearrange channels of the image
    img2 <- c(img[[3]],img[[2]],img[[1]])
   
  # set output path 
    setwd("D:/5_pytorch/pre_processing/1_prediction_data/tiles")
    # compose image file name of current tile
    imgname <- paste0("pred_", i1, "_", i2, ".tif")
    # write tile to harddisc
    writeRaster(img2, filename = imgname)
    }
    
    print(i1)
	  
}