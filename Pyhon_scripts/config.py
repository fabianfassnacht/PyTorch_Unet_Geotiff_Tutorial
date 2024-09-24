# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = 'D:/0_Tutorial/2_training_data'

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "imgs")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# define the test split
TEST_SPLIT = 0.2
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 4
WEIGHT = 0.25
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 3
BATCH_SIZE = 32
# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = 'D:/0_Tutorial/4_output'
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "1_training_output/unet_parakou_50epochs_256px.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "1_training_output/plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "1_training_output/test_paths.txt"])
PRED_PATHS = 'D:/0_Tutorial/4_output/2_predictions'
PRED_PATHS_CONT = 'D:/0_Tutorial/4_output/3_continuos_predictions'