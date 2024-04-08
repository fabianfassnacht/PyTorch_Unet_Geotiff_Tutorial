rm(list = ls())

# load necessaries packages
library(terra)
library(sf)

# !!!!! define pixel size
tile_pix = 200

# get input raster files (matching the extent of the delineated tree polygons)
fils <- list.files("D:/0_Tutorial/0_data/raster/", pattern=".tif$", full.names = T)

# get reference polygons (containing delineated trees)
tree <- vect("D:/0_Tutorial//0_data/shape/Trees.shp")

# start looping through the raster files
for (u in 1:length(fils)){
  
  # import and plot image
  parakou <- rast(fils[u])
  plot(parakou)
  
  # crop reference polygons to extent of raster file
  tree2 = crop(tree, ext(parakou))
  # calculate resolution/pixel size of raster file
  res_ras <- res(parakou)[1]
  
  # get extent of current raster file
  studarea <- ext(parakou)
  # get tile size for clipping (number of pixels times spatial resolution)
  tilesize = tile_pix*res_ras
  
  # build coordinates for tiles
  xsteps <- seq(studarea[1], studarea[2], tilesize)
  ysteps <- seq(studarea[3], studarea[4], tilesize)
  
  

  # loop through tile coordinates  
  for (i1 in 1:(length(xsteps)-1)){
    for (i2 in 1:(length(ysteps)-1)){
      
      # get extent of current tile
      clipext <- ext(xsteps[i1], xsteps[i1+1], ysteps[i2], ysteps[i2+1])
      
      # crop raster file to current extent of tile
      img <- crop(parakou, clipext)
      # change order of channels
      img2 <- c(img[[3]],img[[2]],img[[1]])
      #plotRGB(img, r=3, g=2, b=1)
      mask_dummy <- crop(tree, clipext)
      #plot(mask_dummy)
      # get dummy image for checking for NAs (see below)
      img3 <- img2
      img3[img3==0] <- NA
      
      # if current tile does not contain any reference polygons
      # create an empty mask file
      if (length(mask_dummy) == 0) {
        
        mask <- img[[1]]
        values(mask) <- NA
      
      # if  current file contains reference polygons
      # rasterize them to the mask file
      } else {
        
        mask2 <- img[[1]]
        values(mask2) <- NA
        mask <- rasterize(mask_dummy,mask2)
        mask[is.na(mask)]<-0.0
        plot(mask)
      }
      
         
      # check how many na-pixels the image has and if more than 5% of the image
      # are na-pixels, don't save the image
      if (sum(is.na(values(img3)))>tile_pix*tile_pix*0.05){
        
        print("image dropped")
        next

      } else {
        
        setwd("D:/0_Tutorial\2_training_data")
        imgname <- paste0("img", u, "_", i1, "_", i2, ".tif")
        writeRaster(img2, file=imgname)
        
        setwd("D:/0_Tutorial\2_training_data")
        maskname <- paste0("mask", u,"_", i1, "_", i2, ".tif")
        writeRaster(mask, file = maskname)
        
        }
      
      
    }
    
    print(i1)
    
  }
}
