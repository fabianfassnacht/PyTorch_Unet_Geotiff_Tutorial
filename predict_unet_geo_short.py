# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
#import matplotlib.pyplot as plt
import numpy as np
import torch
#import cv2
import os
from torch import permute
from torchvision.transforms import ToTensor
import rasterio   
from torchvision import transforms
from rasterio.transform import from_origin
    
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		
		with rasterio.open(imagePath) as src:
            # Get the metadata of the source raster
		    meta = src.meta.copy()
		    meta['count'] = 1
    
		image = rasterio.open(imagePath)
		image = image.read()
		image = ToTensor()(image)
		image = permute(image, (1,2,0))
		image = image.unsqueeze(0)
		image = image.cuda()
		print(image.shape)
        

		# find the filename and generate the path to ground truth
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath1 = os.path.join(config.MASK_DATASET_PATH, filename)
		groundTruthPath = groundTruthPath1.replace('img', 'mask')
		predPath1 = filename.replace('img', 'pred')
		predPath = os.path.join(config.BASE_OUTPUT, predPath1)
		print(groundTruthPath)
		print(predPath)

		predMask = model(image).squeeze()
		checkpred = predMask.cpu().numpy()
		print(checkpred)
		#predMask = torch.sigmoid(predMask)
		checkpred = predMask.cpu().numpy()
		print(checkpred)
		#print(predMask.numpy())
		predMask = predMask.cpu().numpy(); predMask
		#cv2.imwrite(predPath, predMask*255)
		with rasterio.open(predPath, 'w', **meta) as dst:
		     #dst.crs = src.crs
		     dst.write(predMask, 1)


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=5)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path) 