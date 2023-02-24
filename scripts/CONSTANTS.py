'''
FOR CONSTANTS USED ACROSS REPOSITORY
'''

'''
Hyperparameters
'''
# initial learning  rate for networks
INIT_LR = 0.002
BATCH_SIZE = 4

'''
Glob vars
'''
# RGB image shape of 128 x 128
IMG_SHAPE = (128, 128, 3)
# dimension size of GAN image output window
DS = 2

# ImageNet dataset directory
IMG_DIR = r'C:\Users\bluet\Downloads\SRA\ILSVRC2013_DET_train'
# MindBigData raw EEG dataset directory
EEG_DIR = r'C:\Users\bluet\Downloads\SRA\MindBigData-Imagenet-IN-v1.0\MindBigData-Imagenet'
# MNIST EEG-image dataset directory
MNIST_EEG_DIR = r'C:\Users\bluet\Downloads\SRA\MindBigDataVisualMnist2022-Cap64v0.016'
