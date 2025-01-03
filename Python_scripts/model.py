# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Dropout
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
	def __init__(self, inChannels, outChannels, dropout=0.5):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3, padding=1)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3, padding=1)
		# Dropout Layer
		self.dropout = Dropout(p=dropout)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		x = self.relu(self.conv1(x))
		x = self.conv2(x)
		x = self.dropout(x)
		return x
        

class Encoder(Module):
	def __init__(self, channels=(3, 4, 8, 12), dropout=0.5):
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
    def __init__(self, channels=(12, 8, 4), dropout=0.4):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        
        # Update: Adjust input channels for concatenation
        self.dec_blocks = ModuleList(
            [Block(channels[i + 1] * 2, channels[i + 1])  # Double input channels
             for i in range(len(channels) - 1)])

        self.conv1x1 = ModuleList(
            [Conv2d(channels[i + 1], channels[i + 1], 1)
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            encFeat = self.conv1x1[i](encFeat)  # Align encoder feature channels
            x = torch.cat([x, encFeat], dim=1)  # Concatenation doubles channels
            x = self.dec_blocks[i](x)  # Pass through the correct block
        return x
    
    # Crop method to align encoder and decoder feature map sizes
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        return CenterCrop([H, W])(encFeatures)



    
class UNet(Module):
    def __init__(self, encChannels=(3, 4, 8, 12),
    	 decChannels=(12, 8, 4),
    	 nbClasses=1, retainDim=True,
    	 outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH),
         dropout=0.5):
    	super().__init__()
    	# initialize the encoder and decoder
    	self.encoder = Encoder(encChannels, dropout)
    	self.decoder = Decoder(decChannels, dropout)
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
	     	#map = F.interpolate(map, self.outSize)
	     	#map = F.interpolate(map, x.shape[2:], mode='bilinear', align_corners=False)
	     	map = F.interpolate(map, self.outSize, mode='bilinear', align_corners=False)
		# return the segmentation map
	    return map