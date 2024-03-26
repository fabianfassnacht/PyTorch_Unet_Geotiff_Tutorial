# remove all current files
rm(list = ls())

# load necessaries packages
library(terra)
library(sf)

# get raster-filenames and paths (for areas covered by training data shapefile)
fils <- list.files("D:/5_pytorch/pre_processing/0_data/raster/", pattern=".tif$", full.names = T)
# load training data shapefile (delineated tree crowns)
tree <- vect("D:/5_pytorch/pre_processing/0_data/shape/Trees.shp")

# start looping through the raster files

for (u in 1:length(fils)){
  
  # import image
  parakou <- rast(fils[u])
  plot(parakou)
  
  ## crop tree shapefile to current raster file
  tree2 = crop(tree, ext(parakou))
  
  # get spatial resolution of current raster file
  res_ras <- res(parakou)[1]
  
  # get extent of current raster file
  studarea <- ext(parakou)
  # !!!!! define tile size (here 128 by 128 pixels)
  tilesize = 128*res_ras
  
  # create vectors defining the regular steps to tile the study area
  xsteps <- seq(studarea[1], studarea[2], tilesize)
  ysteps <- seq(studarea[3], studarea[4], tilesize)
  
  # now loop through the tiles
  for (i1 in 1:(length(xsteps)-1)){
    for (i2 in 1:(length(ysteps)-1)){
      
      # get extent of current tile
      clipext <- ext(xsteps[i1], xsteps[i1+1], ysteps[i2], ysteps[i2+1])
      
      # crop the current raster file to current tile extent
      img <- crop(parakou, clipext)
      # change order of channels
      img2 <- c(img[[3]],img[[2]],img[[1]])
      # crop the reference data to current tile extent
      mask_dummy <- crop(tree, clipext)

      # if the reference data DOES NOT contains any delineated tree polygons      
      if (length(mask_dummy) == 0) {
        
        # take first band of image tile
        mask <- img[[1]]
        # set all pixel to a value of 0
        values(mask) <- 0
      
      # if the reference data DOES contain delineated trees
      } else {
      
      	# take first band of image tile to create the mask
        mask2 <- img[[1]]
        # set all pixel values to 0
        values(mask2) <- 0
        # rasterize the delineated trees to the mask file
        mask <- rasterize(mask_dummy,mask2)
        # set values to 1
        mask[mask==1]<-1
        # plot the mask
        plot(mask)
      }
      
      # save image and mask files to disc
      setwd("D:/5_pytorch/pre_processing/imgs")
      imgname <- paste0("img", u, "_", i1, "_", i2, ".tif")
      writeRaster(img2, file=imgname)
      
      setwd("D:/5_pytorch/pre_processing/masks")
      maskname <- paste0("mask", u,"_", i1, "_", i2, ".tif")
      writeRaster(mask, file = maskname)
      
      
    }
    
    # print current iteration
    print(i1)
    
  }
}
