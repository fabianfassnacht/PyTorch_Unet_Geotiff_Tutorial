## **U-net segmentation of remote sensing data with PyTorch**

**Introduction** 

Hello and welcome to this tutorial which will focus on the "from-scratch" training of a Deep-Learning U-net segmentation model in Python using remote sensing data in the **tif-format** and training data stored as **Polygon-Shapefile** spatially matching the remote sensing data. The tutorial will make use of both R and Python to complete the processing steps. 

It will cover all steps from setting-up a Python environment, to tiling the training data, learning the network and making continuous predictions on the remote sensing image.

The code for the U-net segmentation was adapted from the very helpful Tutorial provided by pyimagesearch that can be found here:

[Original u-net tutorial](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)

The main modifications that are implemented here were motivated by two main points: 1. to make the code work with remote sensing data in which the scaling of the pixel values (which is often automatically occurring when working with for example PNG-files) is often not a sound option (since remote sensing data are not just images but physical measurements of radiance or reflectance) 2. to maintain the geo-coding of the remote sensing data by modifying the code to allow geo-tiff files as input data.

The author of this tutorial is a long-time R-user and new to Python and also to deep learning. So some of the newly introduced Python-code parts may be unnecessarily complicated.  Furthermore, pre- and post-processing steps were mostly implemented in R. So you will have to use both environments. It is assumed that you have R and RStudio already set-up and the packages "terra" and "sf" installed. The work-flow was developed on a Windows 11-Computer with a NVIDIA xxxx GPU and 128 GB RAM.

The python-files required to run the Tutorial can be found here:

[Python codes](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/tree/main/Python_scripts)

While the R-codes can be found here:

[R codes](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/tree/main/R_scripts)

A complete prepared folder structure in a zip-file including training data tiles can be found here:

enter Google Drive Link

Feel free to send me comments if you have suggestions on how to improve the tutorial! fabian [d0t] fassnacht [at] fu-berlin.de

The tutorial is structured as follows:

 - Part 1: Setting-up the Python environment 
 - Part 2: Setting-up a folder structure  
 - Part 3: Overview of the deep-learning work-flow  
 - Part 4: Introduction of the dataset used in the tutorial  
 - Part 5: Pre-processing of the datasets in R    
 - Part 6: Overview of the building blocks of the Python-Workflow and the involved scripts  
 - Part 7: Detailed explanation of the building blocks of the Python-Workflow
 - Part 8: Training of the u-net segmentation  
 - Part 9: Predicting to individual tiles  
 - Part 10: Prediction to continuous areas  
 - Part 11: Post-processing to derive continous predictions maps
 - Part 12: Exploring the results in QGIS
 - Part 13: How to improve the results
 - Part 14: Running the work-flow in Google Colab
 

**Part 1: Setting-up the Python environment and Spyder**

In Python it is common to set-up environments within which the actual coding and development is accomplished. The idea of an environment is that you install the packages and drivers that you need for your work in sort of an " independent copy" of the original Python installation. The advantage of doing this is that you can have various Python versions and combinations of packages and drivers at the same time. This allows you to ensure that a running work-flow is not corrupted by installing a new package or another Python version you need for another work-flow.

This tutorial works with the Anaconda/Miniconda distribution of Python and the set-up of the environment will be described accordingly. As editor we will use Spyder which is delivered with Anaconda/Miniconda. You can download Miniconda here:

[Anaconda download page](https://docs.anaconda.com/free/miniconda/miniconda-install/)

As first step we will create the environment using the Anaconda prompt. You can open the Anaconda prompt by typing "Anaconda" into the windows search bar (Figure 1).

![Figure 1](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_01.png)

in the now opening command line window you will have to execute several commands. **In some cases, it will be necessary to confirm by pressing the "y" button and enter**. You will find the commands that you have to execute below. Only enter the lines of code **without** the leading # - these lines provide some information to better understand the code. 

    # create and activate conda environment using the anaconda 
    # prompt - be aware that we install a specific version of 
    # Python in this case. This is important since providing
    # no information normally leads to the installation of the
    # latest Python version which is often not compatible with
    # some packages.
    # The only thing you may want to adapt here is the name of 
    # the environment - it is "fastai_ff " in my case.
    
    conda create --name fastai_ff python=3.11.5
    
    # as next step we will activate the environment we just 
    # created
    
    conda activate fastai_ff

    # then we will install the necessary packages - besides
    # pytorch - the main package we will use for the deep 
    # learning, we also install fastai and some additional
    # packages which may become handy at a later point in time.
    # this also includes NVIDIA drivers for the case that your 
    # computer has a GPU
    
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c fastai fastai
    
    # besides the deep-learning related libraries we install
    # two additional ones - one that allows us to use Spyder with
    # our environment and another one that we will need later 
    # in the work-flow
    
    conda install spyder-kernels=2.5
    conda install rasterio
    conda install imutils

If you want to check whether your installation was successful, you can run some additional lines of commands within the currently active environment and quickly train a deep learning model. To do this, you can run the following lines:

    # check whether installation worked by running a first model
    
    # start python
    python 

    # import necessary packages
    from fastai.vision.all import *
    from fastai.text.all import *
    from fastai.collab import *
    from fastai.tabular.all import *
    
    # download dataset
    path = untar_data(URLs.PETS)/'images'
    
    # create dataloader
    def is_cat(x): return x[0].isupper()
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224), num_workers=0)
    
    # set up and train the model
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)

If this code runs successfully, your environment should be set-up correctly and we are ready to switch to the Python Editor Spyder.

In Spyder we will now have to make sure that Spyder uses the environment we just created as standard-setting when we open it. For this we open the preferences menu of Spyder by selecting:

**Tools => Preferences** (Figure 2)

![Figure 2](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_02.png)

This should lead to the situation shown in Figure 3. Now we have to select the menu-point **"Python-Interpreter"** on the left (marked in red) and then use the navigation button on the right side (marked in red and with number 1 in Figure 3) to navigate to and select the **"Python.exe"**  executable file located in the Python environment we have just created. 

![Figure 3](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_03.png)

We confirm by clicking "ok" and then restart Spyder by closing the program and re-opening it. If everything went smootly, Spyder should now be connected with the fastai_ff Python environment.

Be aware that your environment and the corresponding Python.exe will most likely be at a different location. In my case, I created the environment in an own folder which is reserved for this ("E:/Python_environments). If you did not make any specifications, your environment was most likely be created in the standard folder for Python-environments which should be specified in the Anaconda Navigator. If you have troubles finding the folder, you may simply want to use the search function of the Windows Explorer and enter the name you gave to your environment.

**Part 2: Setting-up a folder structure**  
Following the original tutorial of pyimagesearch we will set-up a folder stucture that helps us to organize the different python-code files as well as input and output files. For this we will open the Windows Explorer and then use the "right-click" => New => Folder option to create a folder structure that looks as shown in Figure 4.

![Figure 4](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_04a.png)

That means in our main folder (here called "my_unet" - but feel free to name it differently) we create seven subfolders:

**"0_data"**

**"1_prediction_data"**

**"2_training_data"**

**"3_output"**

**"4_predict_comp"**

**"pyimagesearch"**

The subfolder 0_data has two subfolders named 

"raster"
"shape"

The subfolder 1_prediction_data has one subfolder named 

"tiles"

The subfolder 2_training_data has two subfolders named 

"imgs"
"masks"

In these two folders, you will have to store the tiles you create in step X. 

The subfolder 3_output_data has three subfolders named 

"1_training_output"
"2_predictions"
"3_continuos_predictions"
"4_prediction_maps"

The subfolder 4_predict_comp has two subfolders named 

"imgs"
"masks"

From the Python files you downloaded from the link at the start of the Tutorial we will place the three files

**"config .py"
"dataset .py"
"model .py"**

into the folder named "**pyimagesearch**". The two files 

**"predict .py"
"train .py"** 

should be placed in the main folder ("**0_Tutorial**"). Your folder structure should hence look like in Figure 5 in the end.

![Figure 5](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_05a.png)


**Part 3: Overview of the deep-learning work-flow**  
A rough overview over the deep-learning work-flow we will learn today is summarized in Figure 6. The presented principle is somewhat unique to remote sensing data since it involves a tiling-step which is not always needed in deep-learning based image analysis.

![Figure 6](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_06.png)

The whole work-flow is subdivided into seven main steps:

**Acquire remote sensing data** - this typically are very high resolution satellite data or airborne data

**Collect labelled data** - this is a key bottleneck of training any deep learning algorithm from scratch since this step can involve a high level of manual work to mark or delineate the objects of interest in the image data

**Cut images and labels into tiles** - when working with remote sensing data we normally deal with images that cover a comparably large area and have very large x,y-extents (several thousands of pixels). To be able to train the network, we will need to subdivide the subsets of the large image for which we have training data available into smaller image-tiles which will later on each represent one independent training or validation sample. For each of the image-tiles we will need a mask-tile which - in our case - corresponds to a binary-mask which indicates where (in which pixel) in the image-tile the class of interested is present (value of 1) or absent (value of 0). Depending on which algorithm we work with, the mask can also contain more than one class and it would of course then no longer by binary. In ideal case the geo-coding of the original remote sensing image should not be lost during this step.

**Train Convolutional Neural Network (CNN) algorithm** - once we have the tiles prepared, we are ready to train the deep learning algorithm. In this tutorial we will apply a so-called unet which is a classical convolutional neural network that was found to perform well on many segmentation tasks in image analysis. Segmentation refers to the automatic delineation of the class of interest in the remote sensing images as compared to object detection where the object is not delineated but only its location determined and in some cases a bounding box around the object will be provided.

**Apply algorithm on tiles** - once we have trained the network we can predict the algorithm to the entire image. For this we will have to again create tiles as the algorithm can only applied to data that has the same structure as the data it has been trained with. This time we can tile the entire image. Due to some reasons explained further below, it can be recommendable to create overlapping tiles.

**Re-mosaic tiles** - as the last step, we will have to re-mosaic all prediction maps of the tiles but this is a quite straightforward automated process. If we used overlapping tiles, we have to tell the mosaicing function how we want to handle areas where multiple values are available. A straightforward solution could be to simply apply a mean or a max function.

**Apply threshold** - in this tutorial, our unet will provide us continuous values representing the likelihood that each pixel in a tile represents the object of interest. That is, in order to create a prediction mask, we will have to apply a threshold as final step. Finding the optimal threshold can also be automatized using a performance metric such as "Intersection over Union".

This section was meant to give a first idea of the overall work-flow which we hope will be helpful to be able to follow the more detailed processing steps following below.

**Part 4:  Introduction of the dataset used in the tutorial **  

In this tutorial we will make use of a WorldView-3 scene from the cities Parakou in Benin and Berlin in Germany. We will use a pan-sharpened image with three channels (Near-Infrared, Red, Green). An impression of the image quality is given in Figure 07 (left panel). On the right panel, we can see the same image extent but this time overlaid with the training data that we will use in the tutorial (Figure 07 right panel). The training data consists of hand-drawn polygons delineating tree crowns for some sub-parts of the areas covered by the WorldView-3 scene (Figure 08). Both, the WorldView-3 and the shapefile polygon of the dataset from have the coordinate reference system EPSG: 32631 while the data from Berlin has the coordinate reference system EPSG: 32633.

For each of the sub-parts covered with the polygons, a separate subset of the WorldView-3 scene is provided. In the dataset you downloaded these are named: mask_01, mask_02, ..., mask_14. You can find the data in the folder: 

**0_Tutorial/0_data/raster**

Subsetting the images makes the pre-processing slightly easier - an alternative way would have been to subset the file containing the polygons which is named:

Trees.shp and can be found in:

**0_Tutorial/0_data/shape**

These datasets will serve as input to the pre-processing script in the next step during which the image tiles and the masks will be created.

**Part 5: Pre-processing of the datasets in R**  

For creating the image-tiles in tif-format that can be used to train the unet in Python, we will use an R-script. In the script we will use the terra and sf packages. The script is provided below and I hope that the detailed comments in the script will be sufficient to understand what is happening. In order to run the script on your computer, you have to have download the tutorial files provided above and put them in a folder which you are able to find on your PC. In the code below, the image files are stored in the path:

**"D:/0_Tutorial/0_data/raster/"**

while the shapefile is stored in the path:

**"D:/0_Tutorial/0_data/shape"**

You will have to adapt these paths according to where you stored the files on your computer. In the script below - **all section where you have to adapt the code are marked with 5 exclamation marks !!!!! - this principle will be applied throughout the tutorial.**

	# remove all open variables and files
	rm(list = ls())

	# load necessaries packages
	library(terra)
	library(sf)
	
	# !!!!! load the paths to all image-subsets overlapping with the 
	# reference polygons 
	fils <- list.files("D:/0_Tutorial/0_data/raster/", pattern=".tif$", full.names = T)
	
	# !!!!! load the reference polygons
	tree <- vect("D:/0_Tutorial/0_data/shape/Trees.shp")
	## plot tree shapefile to see of it has been loaded correctly (optional)
	plot(tree)

	# start a loop that loads one image-subset after the other
	for (u in 1:length(fils)){
	  
	  # import image-subset
	  parakou <- rast(fils[u])
	  # plot image to see of it has been loaded correctly (optional)
	  plot(parakou)
	  
	  # crop the extent of the file containing the reference polygons
	  # to the extent of the current image subset	  
	  tree2 = crop(tree, ext(parakou))
	  
	  # obtain the pixel size of the current image subset
	  res_ras <- res(parakou)[1]
	  
	  # get the geographic extent (xmin, xmas, ymin, ymax) of the 
	  # current image subset	  
	  studarea <- ext(parakou)
	  
	  # !!!!! define the tile size
	  # the tile size will define how large the tiles you will feed into
	  # the deep learning algorithm are. The tile size does have an effect on
	  # the way the algorithm learns and is hence one of the parameters you can
	  # adjust. the larger the value you chose, the larger the tiles will be 
	  # and the less tiles you will create. In this example the tile size would be
	  # 128 by 128 pixels
	  tilesize = 128*res_ras
	  
	  # create the corder-coordinates of the tiles
	  xsteps <- seq(studarea[1], studarea[2], tilesize)
	  ysteps <- seq(studarea[3], studarea[4], tilesize)
	  
	  # start a double loop to clip the current sub-image into
	  # equally sized, non-overlapping tiles 
	  for (i1 in 1:(length(xsteps)-1)){
	    for (i2 in 1:(length(ysteps)-1)){
	      
				# compose the extent of the current tile	      
	      clipext <- ext(xsteps[i1], xsteps[i1+1], ysteps[i2], ysteps[i2+1])
	      
	      # crop the image to the current tile extent
	      img <- crop(parakou, clipext)
	      # rearrange the order of the channels
	      img2 <- c(img[[3]],img[[2]],img[[1]])
	      
	      # crop the reference polygon file to the extent of current tile
	      mask_dummy <- crop(tree, clipext)
	         
	      # check if the cropped polygon file contains any polygons
	      # if this is not the case:   
	      if (length(mask_dummy) == 0) {
	        
	        # copy one band of the current tile
	        mask <- img[[1]]
	        # set all values of the band to 0 (no reference polygon)
	        values(mask) <- 0
	        
	      # if it is the case   
	      } else {
	        
	        # copy one band of the current tile
	        mask2 <- img[[1]]
	        # set all values of the band to 0 
	        values(mask2) <- 0
	        # then rasterie the polygon objects into the 
	        # raster file
	        mask <- rasterize(mask_dummy,mask2)
	        mask[mask==1]<-1
	        plot(mask)
	      }
	      
	      # !!!!! set the output path for the image tiles
	      setwd("D:/0_Tutorial/2_training_data/imgs")
	      # compile an output filename
	      imgname <- paste0("img", u, "_", i1, "_", i2, ".tif")
	      # save the file to the harddisc
	      writeRaster(img2, file=imgname)
	      
	      # !!!!! set the output path for the mask tiles
	      setwd("D:/0_Tutorial/2_training_data/masks")
	      # compile an output filename
	      maskname <- paste0("mask", u,"_", i1, "_", i2, ".tif")
	      # save the file to the harddisc
	      writeRaster(mask, file = maskname)
	      	      
	    }
	    # print id of current iteration of current image-subset
	    print(i1)
    }
	}


If everything runs smoothly, this processing step will take a while and you should end up with a situation as shown in Figure 9, that is a folder containing the image tiles and one folder containing the corresponding mask files. The two folder should have the same amount of files and order of files. Otherwise, the mask-files are not correctly linked to the image files in later steps of the tutorial. 
 
**CAREFUL**: In many cases a .Rhistory file is saved in the masks folder. **Please delete this file** to make sure that the images and masks are correctly assigned to each other in later parts of the work-flow.
 
Insert Figure 09

**Part 6: Overview of the building blocks of the Python-Workflow and the involved scripts** 

After pre-processing the input files we are now able to start working on the Python scripts with which we will train (or "learn") the neural network. The Python scripts are developed to work with a GPU and in ideal case your computer has a NVIDIA graphic card with a GPU that is supported by PyTorch. The script may also work with a CPU but it may take notably longer and you might also have to modify some lines of codes to convert the format of the tensors in a CPU-readable format. We will not go into details here how this can be done since even if you have no GPU on your computer, you have the option to run the code in the Google Colab environment where GPUs are freely available (at least for training smaller models). At the end of the tutorial, I will provide some instructions on how you can run the work-flow in the Google Colab environment.

The Python work-flow includes six Python-code-files in total. We will give a brief overview here and then provide and discuss the code of each file with more details below.

There are are three files that you will run in the Python environment step-by-step, these include the files "train.py", "predict_unet.py" and "predict_unet_geo_areawide.py". The latter two are almost identical with the first one allowing to make a prediction of the trained model to a small subset of the hold-out samples (individual tiles, spread over the whole area of interest) and the second one for predicting the trained model to the whole or continuous subsets of the image.

The "train.py" file will call three files (config.py, "dataset_tif.py and "model.py") which are the **building blocks** of the work-flow to train the neural network. It is important that the code of these three files are not integrated into "train.py" as this will allow us to initiate parallel runs on the GPU.  The "config.py" file contains numerous settings that you might want to adapt and change to improve the results of your model. For example the number of epochs, the size of the dataset that is reserved for testing and the batch size can be defined here. The "model.py" files contains the architecture of the unet-model. This is the one file which you will most likely not change at all in the context of this tutorial. Finally, the "dataset_tif.py" contains code that pre-processes the image-tiles you use as input to prepare them in a format that can be understood by the deep-learning algorithm. It basically loads the image-tiles and transforms them to tensors (multi-dimensional matrices) that can read by the GPU. The term **"dataloader"** is often used in this context.

In the following we will now have a closer look at each of the three building block files and explain with more details what they are about.

**Part 7: Detailed explanation of the building blocks of the Python-Workflow**

!!!!! Be aware that if you copy the code below instead of using the provided files, you might end up having to fix the intends (distances from the left side of your text-editor) in your Python editor since sometimes the formatting is not preserved when copying the files into MarkDown and intends have a meaning in Python .

***config.py***

The config.py file is comparably short but includes many important parameters that will be called when training the neural network. In the following you will see the code with corresponding detailed explanations in the comments. Again a reminder: the five exclamation marks !!!!! remind you of the parts in the code which you either have to or alternatively can adapt:

	# import the necessary packages
	import torch
	import os

	# !!!!! set the path to the folder which contains the
	# two sub-folders with image files and masks
	DATASET_PATH = 'D:/5_pytorch/dataset/train'

	# !!!!! define the path to the images and masks sub-folders
	IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "imgs")
	MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
	# !!!!! define the test split (the fraction of images that)
	# will be reserved for testing the model
	TEST_SPLIT = 0.25
	# determine the device to be used for training and evaluation
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	# determine if we will be pinning memory during data loading
	PIN_MEMORY = True if DEVICE == "cuda" else False
	# define the number of channels in the input, number of classes,
	# and number of levels in the U-Net model. For our tutorial
	# these settings should be fine - but you might want to adapt
	# these if you work with another dataset
	NUM_CHANNELS = 3
	NUM_CLASSES = 1
	NUM_LEVELS = 3
	# !!!!! initialize learning rate, number of epochs to 
	# train for and the batch size
	INIT_LR = 0.001
	NUM_EPOCHS = 50
	BATCH_SIZE = 64
	# !!!!! define the input image dimensions
	INPUT_IMAGE_WIDTH = 128
	INPUT_IMAGE_HEIGHT = 128
	# define threshold to filter weak predictions
	# in this tutorial we have deactivated the treshold 
	# and produce continuous outputs instead
	THRESHOLD = 0.5
	# !!!!! define the path to the base output directory
	BASE_OUTPUT = 'D:/5_pytorch/output'
	# !!!! define the path to the output serialized model
	# model training plot, and testing image paths
	MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_parakou.pth")
	PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
	TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
	PRED_PATHS = 'D:/0_Tutorial/4_output/2_predictions'
	PRED_PATHS_CONT = 'D:/0_Tutorial/4_output/3_continuos_predictions'

***dataset_tif.py***

The dataset_tif.py file contains the data-loader and hence the work-flow to load the image-tiles in tif-format and then transform them to a GPU-tensor. For this a user-defined "class" named "SegmentationDataset" is created. 

	# import the necessary packages
	from torch.utils.data import Dataset
	from torch import permute
	from torch import nan_to_num
	from torchvision.transforms import ToTensor
	import rasterio
	
	# define class to load the images
	class SegmentationDataset(Dataset):
		def __init__(self, imagePaths, maskPaths, transforms):
			# store the image and mask filepaths, and augmentation
			# transforms
			self.imagePaths = imagePaths
			self.maskPaths = maskPaths
			self.transforms = transforms
		def __len__(self):
			# return the number of total samples contained in the dataset
			return len(self.imagePaths)
		def __getitem__(self, idx):
			# grab the image path from the current index
			imagePath = self.imagePaths[idx]
		
		# load images using rasterio and convert them
		# to tensors
		image = rasterio.open(imagePath)
		image = image.read()
		image = ToTensor()(image)
		# change order of dimensions of tensor to have 
		# number of bands first, then height and width
		image = permute(image, (1,2,0))
		
		# load mask files and transform to tensor		
		mask = rasterio.open(self.maskPaths[idx])
		mask = mask.read()
		mask = ToTensor()(mask)
		# replace nan-values to 0
		nan_to_num(mask, nan=0.0)
		# change order of dimensions of tensor to have 
		# number of bands first, then height and width
		mask = permute(mask, (1,2,0))
		        
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)


**model.py**

The model.py file contains the neural network model architecture. It is recommended that only advanced users with a good understanding of the individual elements of a neural network should change these lines of code. As you will see the "config.py" file is also called in this file to provide the information on the size of the image and mask tiles.


	# import the necessary packages
	from . import config
	from torch.nn import ConvTranspose2d
	from torch.nn import Conv2d
	from torch.nn import MaxPool2d
	from torch.nn import Module
	from torch.nn import ModuleList
	from torch.nn import ReLU
	from torchvision.transforms import CenterCrop
	from torch.nn import functional as F
	import torch

	# in this section the individual building blocks of the CNN
	# are created
	class Block(Module):
		def __init__(self, inChannels, outChannels):
			super().__init__()
			# store the convolution and RELU layers
			self.conv1 = Conv2d(inChannels, outChannels, 3)
			self.relu = ReLU()
			self.conv2 = Conv2d(outChannels, outChannels, 3)
		def forward(self, x):
			# apply CONV => RELU => CONV block to the inputs and return it
			return self.conv2(self.relu(self.conv1(x)))
	        

	class Encoder(Module):
		def __init__(self, channels=(3, 16, 32, 64)):
			super().__init__()
			# store the encoder blocks and maxpooling layer
			self.encBlocks = ModuleList(
				[Block(channels[i], channels[i + 1])
				 	for i in range(len(channels) - 1)])
			self.pool = MaxPool2d(2)
        
		def forward(self, x):
			# initialize an empty list to store the intermediate outputs
			blockOutputs = []
			# loop through the encoder blocks
			for block in self.encBlocks:
				# pass the inputs through the current encoder block, store
				# the outputs, and then apply maxpooling on the output
				x = block(x)
				blockOutputs.append(x)
				x = self.pool(x)
			# return the list containing the intermediate outputs
			return blockOutputs

	class Decoder(Module):
		def __init__(self, channels=(64, 32, 16)):
			super().__init__()
			# initialize the number of channels, upsampler blocks, and
			# decoder blocks
			self.channels = channels
			self.upconvs = ModuleList(
				[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
				 	for i in range(len(channels) - 1)])
			self.dec_blocks = ModuleList(
				[Block(channels[i], channels[i + 1])
				 	for i in range(len(channels) - 1)])
        
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
    
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures
  
	# in this section the building blocks are combined to form
	# the unet-architecture   
	class UNet(Module):
	    def __init__(self, encChannels=(3, 16, 32, 64),
	    	 decChannels=(64, 32, 16),
	    	 nbClasses=1, retainDim=True,
	    	 outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
	    	super().__init__()
	    	# initialize the encoder and decoder
	    	self.encoder = Encoder(encChannels)
	    	self.decoder = Decoder(decChannels)
	    	# initialize the regression head and store the class variables
	    	self.head = Conv2d(decChannels[-1], nbClasses, 1)
	    	self.retainDim = retainDim
	    	self.outSize = outSize
	    def forward(self, x):
			# grab the features from the encoder
		    encFeatures = self.encoder(x)
			# pass the encoder features through decoder making sure that
			# their dimensions are suited for concatenation
		    decFeatures = self.decoder(encFeatures[::-1][0],
				encFeatures[::-1][1:])
			# pass the decoder features through the regression head to
			# obtain the segmentation mask
		    map = self.head(decFeatures)
			# check to see if we are retaining the original output
			# dimensions and if so, then resize the output to match them
		    if self.retainDim:
		     	map = F.interpolate(map, self.outSize)
			# return the segmentation map
		    return map

 **Part 8: Training of the u-net segmentation**  
	
After getting to know the building-block python files which should all be stored in the folder named "pyimagesearch" (see section 2) we are now ready to train our unet deep learning model. In the code below, it is assumed that you have created the exact folder structure as shown in section 2 and that this folder structure is stored in the path "D:/0_Tutorial". You will most likely have to adapt the code below to the path on your computer where you have created the folder structure. It is recommended to run the individual code-blocks shown below one after the other (the individual code blocks are always interrupted by some explanations). The code below is the code that you will find in the "train.py" file on github.

	# import the necessary packages and set directory
	import os
	# !!!!! change current directoy to the folder in which the folder structure
	# of section 2 can be found on your pc
	os.chdir("D:/0_Tutorial")
	
	# be aware that you are here loading the building blocks from the 
	# other python files
	from pyimagesearch.dataset import SegmentationDataset
	from pyimagesearch.model import UNet
	from pyimagesearch import config
	
	# starting from here we load several pytorch libraries 
	# and other libraries we will need in the following
	from torch.nn import BCEWithLogitsLoss
	from torch.optim import Adam
	from torch.utils.data import DataLoader
	from sklearn.model_selection import train_test_split
	from torchvision import transforms
	from imutils import paths
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import torch
	import time


Be aware that if you receive error messages when running the code above, you might have forgotten to install some of the required packages and you might have to go back to Step 1 and install the corresponding packages in your Python environment. After all the packages have been successfully loaded, we can proceed with initiating the data loader.

	
	# load the image and mask filepaths in a sorted manner
	# as you can see the paths to the images are being called 
	# from the config.py file - so if this step fails, you might have
	# to check your config.py file-
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
	# partition the data into training and testing splits based on
	# the fraction defined in the config file
	split = train_test_split(imagePaths, maskPaths,
		test_size=config.TEST_SPLIT, random_state=42)
	# unpack the data split
	(trainImages, testImages) = split[:2]
	(trainMasks, testMasks) = split[2:]
	# write the testing image paths to disk so that we can use then
	# when evaluating/testing our model
	print("[INFO] saving testing image paths...")
	f = open(config.TEST_PATHS, "w")
	f.write("\n".join(testImages))
	f.close()

As next step we define data transformation and set-up the dataloaders. Data transformations are on the one hand side being used to apply data augmentation (modify the input data slightly to increase the dataset and potentially improve the performance of the neural network on unseen data) but on the other side also to standardize the input images. In our case we apply a "resizing" operation, that is we make sure that all the input image tensors have exactly the same size and we also apply simple data augmentation (flipping and mirroring the image tiles). As we already made sure during the creation of the tiles (Step X) that each tile has the same size, this resizing step is not absolutely necessary in our case, but it can be important with other datasets.

	# define transformations
	transforms = transforms.Compose([transforms.ToPILImage(),
	 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
		transforms.ToTensor()])
	# create the train and test datasets
	trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
		transforms=transforms)
	testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
	    transforms=transforms)
	print(f"[INFO] found {len(trainDS)} examples in the training set...")
	print(f"[INFO] found {len(testDS)} examples in the test set...")
	# create the training and test data loaders
	trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count())
	testLoader = DataLoader(testDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count())

We are now almost ready to train our model. To be able to do this, we have to initiate our model which is stored in the model.py. We do this by running the code below. Be aware that we again call the config.py file to obtain several settings to set-up our model. 

	# Assuming you have a binary segmentation task with 
	# foreground and background
	# pos_weight should be calculated based on the class 
	# imbalance in your dataset
	# For example, if foreground/background ratio is 1:4, 
	# pos_weight should be 1/4 = 0.25 - be aware that the 
	# tensor is also transformed to a cuda-tensor here to 
	# make the code work
	pos_weight = torch.tensor([0.25]).cuda()
	# initialize our UNet model
	unet = UNet().to(config.DEVICE)
	# initialize loss function and optimizer
	lossFunc = BCEWithLogitsLoss(pos_weight = pos_weight)
	opt = Adam(unet.parameters(), lr=config.INIT_LR)
	# calculate steps per epoch for training and test set
	trainSteps = len(trainDS) // config.BATCH_SIZE
	testSteps = len(testDS) // config.BATCH_SIZE
	# initialize a dictionary to store training history
	H = {"train_loss": [], "test_loss": []}

One important parameter to consider while initiating the model is whether we want to apply positive weights or not. This is important if the target class covers notably more or notably less of the images than the background. If we apply a positive weight < 1, we will give more weight to target class when calculating the loss function. This is important since if we for example assume that our target class only makes up 10% of the whole image while the background covers 90%, then classifying the whole image as background would give us an accuracy of 90% but we would still have a completely useless output. So playing around with the positive weights may have quite a strong effect on the model performance and may be an interesting way to improve the results.

If the model is successfully initialized, we can start the actual training process with the code below. The core-part of the code consists of a loop where each iteration of the loop corresponds to an epoch of training of the model. An epoch ends after the training process has processed all image tiles of the training dataset once to update the weights of the neural network. The image tiles are not processed all together at once but in subgroups (batches) of a user-defined batch-size (which has to be defined in the config.py file). This process can take quite a long time, depending on what kind of hardware you are using and which tile-size (and hence sample size) you defined. Python will constantly update on the process by telling you how many epochs are completed and how the test and the train loss develops (Figure X).

	# loop over epochs
	print("[INFO] training the network...")
	# store time when training starts
	startTime = time.time()
	for e in tqdm(range(config.NUM_EPOCHS)):
		# set the model in training mode
		unet.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalTestLoss = 0
		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# perform a forward pass and calculate the training loss
			pred = unet(x)
			loss = lossFunc(pred, y)
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			unet.eval()
			# loop over the validation set
			for (x, y) in testLoader:
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
				# make the predictions and calculate the validation loss
				pred = unet(x)
				totalTestLoss += lossFunc(pred, y)
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps
		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Test loss: {:.4f}".format(
			avgTrainLoss, avgTestLoss))
	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))


Insert Figure X here.

After, the model training has ended we can have a look at the development of the training and test loss of the epochs. The plot resulting from the code below should look like the one shown in Figure X.
	
	# plot the training loss
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["test_loss"], label="test_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	# the output location of the path is also taken from the config.py file
	plt.savefig(config.PLOT_PATH)
	# save the model to disk (it can be loaded later on for predicting)
	torch.save(unet, config.MODEL_PATH)


Insert Figure X here.


**Part 9: Predicting to individual tiles**  

Once our model has been trained successfully, we can now use it to make predictions using the **predict_unet.py** file. For a first check of the quality of the model, it can be useful to make a quick prediction to some of the test tiles that were created by the datasplit at the beginning of the work-flow. The code to make predictions will be nearly the same, independent if we want to predict to few individual tiles or to a large number of continuous tiles to derive a continuous prediction map. The first part of the code is always the same and looks like this:

	# import the necessary packages
	# we will again need some information from the config.py
	from pyimagesearch import config
	# as well as various other packages
	import matplotlib.pyplot as plt
	import numpy as np
	import torch
	import cv2
	import os
	from torch import permute
	from torchvision.transforms import ToTensor
	import rasterio   
	from torchvision import transforms
	from rasterio.transform import from_origin
	
	# the core of the prediction work-flow is the following function:    
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
	    
			image = rasterio.open(imagePath)
			image = image.read()
			image = ToTensor()(image)
			image = permute(image, (1,0,2))
			image = image.unsqueeze(0)
			image = image.cuda()
			print(image.shape)
	        

			# get the filename and poutput path and generate the complete path 
			filename = imagePath.split(os.path.sep)[-1]
			groundTruthPath1 = os.path.join(config.MASK_DATASET_PATH, filename)
			groundTruthPath = groundTruthPath1.replace('img', 'mask')
			predPath1 = filename.replace('img', 'pred')
			predPath = os.path.join(config.BASE_OUTPUT, predPath1)
			print(groundTruthPath)
			print(predPath)
	    
	    # apply the model to the input image        
			predMask = model(image).squeeze()
			predMask = predMask.cpu().numpy(); predMask
			with rasterio.open(predPath, 'w', **meta) as dst:
			   dst.write(predMask, 1)

Once the function is defined we can apply it to a random subset of the test tiles available in our dataset. The files will be stored to the output folder defined in the config.py file.

	# load the image paths in our testing file and randomly select 10
	# image paths
	print("[INFO] loading up test image paths...")
	imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
	# !!!!! by changing the "size" parameter, you can decide
	# for how many random image tiles from the test data you
	# want to create a prediction
	imagePaths = np.random.choice(imagePaths, size=2)
	# load our model from disk and flash it to the current device
	print("[INFO] load up model...")
	unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
	# iterate over the randomly selected test image paths
	for path in imagePaths:
		# make predictions and visualize the results
		make_predictions(unet, path) 

The prediction files created with this work-flow contain continuous values that represent the probability that a certain pixel belongs to the target class (in our case trees). We can open the created prediction tiles in QGIS and compare them to original image tiles to see if the predictions are reasonable. It might be necessary to adapt the visualization settings to get a clearer view of the quality of the image. When loading the prediction images, QGIS will most likely not detect the coordinate reference system, but if you set it to EPSG: 32631 by clicking the button marked in red in Figure X, the image tiles should have a correct geolocation and be overlapping with the original image tiles. Some examples of prediction tiles and a comparison to the original image of the model-run described above are shown in Figure X.

**Part 10: Prediction to continuous areas**  

Once we have verified with some of the test tiles that our trained unet is making plausible predictions, we can apply it to a larger image subset which was not used during the training phase at all. In this tutorial, the corresponding file is named "pansharp_parakou_wv3_prediction_area.tif". Similarly as with the datasets used for training, we will now split this image into tiles of the size 256 by 256 pixels (or another tile size that corresponds to the tile size of the training data in case you decided to change it in the upper parts of the tutorial). The code to accomplish the tiling is similar to the code we used above with one small difference - for the continuous prediction areas, we will not produce tiles that lay directly next to each other but we will produce tiles that overlap by 90%. The main reason for this is that we want to avoid edge effects which can occur in unet predictions. If we have a 90% overlap, most of the pixels will receive 9 predictions in total and we can later summarize these predictions into a single value for each pixel and the corresponding summarized image mosaic should have a more homogeneous appearance.

The code to accomplish this step is provided in R and looks like this:

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
	
	# define steps in a way that tiles will overlap 
	xsteps <- seq(studarea[1], studarea[2], (tilesize/9))
	ysteps <- seq(studarea[3], studarea[4], (tilesize/9))

	# loop through the steps and create tiles
	for (i1 in 1:(length(xsteps)-9)){
	  for (i2 in 1:(length(ysteps)-9)){
	    
	    # get extent for current tile
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

This code will now run for quite a bit and you will end up with a huge number of tiles. These tiles will in the next step be used as input to the trained-unet algorithm. This will require only a small modification of the prediction code we already applied above. The first part of the code, the main prediction function, is exactly the same as used above. The lower part, where the predict function is called is changed as seen below:

	    
	# load the image paths in our testing file and randomly select 10
	# image paths
	from imutils import paths
	print("[INFO] loading up test image paths...")
	# !!!!! here you will have to insert the path to the prediction
	# tiles that you have just created
	imagePaths = sorted(list(paths.list_images('D:/5_pytorch/pre_processing/1_prediction_data/tiles')))

	# load our model from disk and flash it to the current device
	print("[INFO] load up model...")
	unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
	# iterate over the randomly selected test image paths
	for path in imagePaths:
		# make predictions and visualize the results
		make_predictions(unet, path) 
 

This step should now require a bit of time to be accomplished as you have created numerous overlapping files in the preceeding steps - in my case these would be more than 63000 files. The predicted files will all be stored in the output folder defined in the "config.py" file.

**Part 11: Post-processing to derive continous predictions maps**

Once all the prediction maps are successfully created, we use a final R-script to merge all of the predicted tiles into a single continuous prediction map. Make sure that you only use the prediction maps from the continuous tiles as input to this R-script.
	
	# load necessary package
	require(terra)

	# get file names and path to all predicted tiles
	fils <- list.files("D:/5_pytorch/output", pattern=".tif$", full.names = T)
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
	fils2$fun <- "max"

	# call mosaic function
	img_end <- do.call(mosaic, fils2)
	# see result
	plot(img_end) 
	# save resulting mosaic to file
	setwd("D:/5_pytorch/post_processing/")
	writeRaster(img_end, filename = "predictions_128_tiles_25epochs.tif")


	# optionally you can apply a threshold to create a binary mask
	# this can be for example done after manually checking the resulting
	# prediction map in QGIS and identifying a meaningful threshold
	img_end_th1 <- img_end > -1300
	# save binary mask to file
	writeRaster(img_end_th1, filename = "predictions_20epochs_gt_m1300.tif")


**Part 12:  Exploring the results in QGIS**

Once you have created the output files (which will be geotiffs as well) you can load them in QGIS along with the original image files and compare the prediction maps with the original image files. In ideal case, all pixels in the image representing trees should have higher values than the other classes in the prediction maps. Figure XX gives an example of a prediction map where the trees were detected fairly well (as seen when comparing the prediction map with the corresponding image tile). On the right hand, we can see a binary mask that was derived from the prediction map by applying a threshold.

Insert Figure XX

**Part 13: How to improve the results**

There are various parameters which you can modify to improve the results of your model. Some of the most common ones are the following:

1. Increase the number of training data (always helpful)
2. Change the tile size
3. Change the number of epochs
4. Adapt the positive weights of the loss function BCEWithLogitsLoss (the value you chose very much depends on your dataset and how frequent the target class is against the background)
5. Change the loss function (no example provided but in the given example, something like "Intersection over Union" may lead to better results)
6. In some cases, your mask files may wrongly prepared and the model fails to learn something - I had some corresponding troubles with nan-values and 0 values. The current code should account for this by transforming nan-values to 0 but you might still want to double-check in case you do not get meaningful results.
7. You can adapt the kernel size of the unet - this is in our model done by modifying the following lines of the model.py script:

	class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))

To change the kernel size from 3 x 3 pixels to for example 11 x 11 pixels you have to change the functions "Conv2D(inChannels, outChannels, 3)" to "Conv2D(inChannels, outChannels, 11)"

    
9. Finally there are options to use pre-trained models and other model architectures, but these may of course require notably different code and major adaptations.
   

One important advice:

To really understand whether your model is improving our not, the loss-values may in our example not be a good indicator since they for example depend on the applied positive weights and are hence not directly comparable between models trained with different positive weight settings.

From my experience, it is best to also have a look at the prediction maps and check whether the observed patterns are actually meaningful or not. In part 15 below, I provide some indications how you can make meaningful graphical model comparisons by keeping a complete hold-out set and make predictions of various trained models always on the same hold out set (never used during model training).

**Part 14: Running the work-flow in Google Colab**

For running the work-flow of this tutorial in Google Colab, you will need a Google account. If you have a Google account, you can automatically use the base version of Google Colab which is free of cost and which allows you to use computing resources (including GPUs) free of charge (at least if no longer run-times are required).

The procedure to implement the whole tutorial on Google Colab is as follows:

1. Run the pre-processing to create the image tiles and masks on your local computer
2. Prepare the folder structure as explained above (with image tiles and corresponding mask files already placed in their folders)
3. Upload the whole folder structure including the building-block Python-code files to your google drive and remember to which paths you have uploaded it.
4. Then open Google Colab using this link: [Google Colab start page](https://colab.google/) 
5. Open a new notebook
6. In this notebook you can copy the code-lines of the train.py file but you will have to add a few lines of code above to mount your google drive:
		
		# mount Google drive to access files of the Tutorial
		from google.colab import drive
		drive.mount('/content/drive/')

	Once you run this code, Google will ask you to grant access to your Google drive and you will have to accept this to proceed.

	Since you are now in a new Python environment (and no longer in the one we prepared at the beginning of the Tutorial) you will have also install some packages:

		!pip install rasterio
		!pip install imutils

	The packages will then be installed within the Python environment of Google Colab. You don't have to install Pytorch and some other packages as they are already pre-installed in the Colab-environment.

	Then, the only thing you still have to do is to adapt the paths, both in the code of the train.py file that you copy&pasted to the Colab notebook and also in the config.py that you have uploaded to your google drive. You might have to delete the file and re-upload it after updating the paths. 

	As an example - if you have uploaded the folder structure to a folder named "0_Tutorial_Colab" which is directly positioned in the base-folder of your google drive, the corresponding code to switch the current path of the Python session to this folder will be:

		# import the necessary packages
		import os
		os.chdir('/content/drive/MyDrive/0_Tutorial_Colab/')

7. One more important advice: The standard Colab session will be using a run-time with only a CPU - if you want to run the work-flow of this Tutorial you will have to use a GPU run-time. You can do this by clicking "connection" (Figure XX - 1) on the top right and then "change run-time"  (Figure XX - 2) and in the now appearing window select the "GPU" option. Be aware that in the cost-free version of Colab, the access to GPUs is limited and you might only be able to get access at certain time-points and for limited hours.

Insert Figure XX

8. Once you have trained the model successfully, you can either download the model (this is the file with an ending of .pth) and then proceed offline using your own computer or by also copy & pasting the code in the "03_predict.. .py" file to the Notebook. You just have to make sure that you indeed updated all relevant paths. The predicted files will be saved to the output folder in your folder structure on the Google Drive. 

**Part 15: Trouble-shooting**

While developing this tutorial I came across various issues that prevented the code from running. Some of them quite plausible, others hard to understand. In case you have troubles getting to the code to run, here are some potential problems which you can work on:

1. If the model starts training but gets stuck in the first epoch and is not making any progress for several hours, it is likely that the hardware of your computer is not sufficient to run the code. Interestingly, in my case this was sometimes hard to explain since neither the PC's memory nor the GPU memory were maxed out. One example was the following: If I switch the tile size from 256 pixels to 128 pixels (and thereby quadruple the amount of samples available for training) the actual size of the dataset is not increased, but the sample size is. If I left all other settings unchanged, the model did not start learning on my small workstation with a NVIDIA xxx GPU. When uploading the data to Google Colab, the code ran with exactly the same inputs.
2. Another odd issue I observed and which occurred arbitrarily was that code the building block codes (dataset_tif.py, config.py and model.py) could not be loaded in the moment when Python splits the work-tasks to multiple jobs. The error message was that the corresponding package/module cannot be found. In my case, the problem was often fixed by simply renaming the folder in which the building block code-files were stored and adapting the corresponding codes of line which load the files:

		# be aware that you are here loading the building blocks from the 
		# other python files
		from pyimagesearch.dataset import SegmentationDataset
		from pyimagesearch.model import UNet
		from pyimagesearch import config

So in this example, the files are saved in a folder named "pyimagesearch" - if I got the corresponding error, I renamed the folder to for example "pybblocks" and adapted the code accordingly:

		# be aware that you are here loading the building blocks from the 
		# other python files
		from pybblocks.dataset import SegmentationDataset
		from pybblocks.model import UNet
		from pybblocks import config

This makes absolutely no sense to me, but fixed the problem most of the time.

3. It is important to always remember that Pytorch uses different tensors for processing the data on a CPU and a GPU. The whole tutorial is written for a GPU and will most likely not work on a machine that only has a CPU. If you do not have a GPU on your local machine, consider using for example Google Colab.

4. The scaling of the input images has an effect on the continuous prediction maps of the output images. In my case I tried out several cases including:
- original Worldview-3 values (most pixel values ranging around 0-800)
- original values divided by 10000 
- original values divided by 2000

With the original values, the output predictions look quite smooth

When dividing the original values by 10000 the continuous prediction maps look quite artificial and distorted (showing some sort of gridded structures)

With dividing the original values by 2000 the continuous predictions maps look relatively smooth again but a little less smooth than the original values




**Part 15: Visual model comparisons using a complete hold-out set** 


