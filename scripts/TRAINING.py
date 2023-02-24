'''
DATA PREPROCESSING AND CVAE-GAN TRAINING
'''

# IMPORTS

# ML LIBRARIES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import mne
from keras import backend as K

# STD LIBRARIES
import os
import time
from os import listdir
import sys
import pickle
import gc
import cProfile
import pstats

# FILE IMPORTS
from CVAE_GAN import CVAE_GAN
from CONSTANTS import *


# GPU Configs: check if GPU is available from CUDA library
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# utilizes memory from one GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# extra check for GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


def find_csv_filenames(path_to_dir, suffix=".csv"):
    '''
    Obtain all the EEG data .csv file names
    '''
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def defineOutputs():
    '''
    Obtain all image categories
    '''
    d = {}
    with open("dataDict.txt") as f:
        for line in f:
           (key, val) = line.split(',')
           d[key] = int(val)
    return d

def generate_data():
    '''
    Preprocessing for image, EEG, and label data from MindBigData and Imagenet datasets
    Data samples must meet the two criteria: images must be RGB and EEG data must last more than or equal to 384 Hz

    precondition: files must exist in directory
    postcondition: returns 3 arrays of EEG, image, and label data respectively
    ''' 
    # DIMENSIONS
    FILE_DIM = 13175
    # FREQ = 1
    FREQ = 1
    CHANNELS = 5 * FREQ
    SAMPLES = 384

    # find all EEG data file names and how many there are
    filenames = find_csv_filenames(EEG_DIR)
    amt_files = len(filenames)

    outputstageDict = defineOutputs()
    
    #  preset EEG data and respective label array sizes
    X = np.zeros((FILE_DIM, CHANNELS, SAMPLES))
    y = np.zeros((FILE_DIM, len(outputstageDict)))

    # preset image data array size
    X_r = np.zeros((FILE_DIM, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))

    for j in range(amt_files):
        # obtain current filename of EEG .csv file
        curr_file = filenames[j]

        # IMG CATEGORY: will always start with "n0..."
        key = filenames[j][filenames[j].find('n0'):filenames[j].find('n0') + 9]
        # IMG NUMBER INDEX
        index = curr_file.find('_')
        index = curr_file.find('_', index+1)
        start_index = curr_file.find('_', index+1)
        ending_index = curr_file.find('_', start_index+1)
        idx = curr_file.find('_', ending_index) + 1
        end_idx = curr_file.find('_', idx)
        num = curr_file[idx:end_idx]

        # IMG file path
        curr_file_r = os.path.join(IMG_DIR, key, key, key) + '_' + num + '.JPEG'

        # obtain current EEG file and their values strictly
        df = pd.read_csv(EEG_DIR + '\\' + curr_file, sep = ',', header = None)
        df = df.values

        # match images to EEG data and save them in their respective Numpy arrays
        if os.path.exists(curr_file_r):
            img = plt.imread(curr_file_r, 0)

            # cut out any any EEG samples less than 384 size and any trials with images being non-RGB
            if img.ndim == 3 and df.shape[1] > 384:
                # normalize from 0 to 1 floats and resize images to 128 x 128 
                X_r[i] = cv2.resize(np.divide(img, 255.0), (IMG_SHAPE[0], IMG_SHAPE[1]))

                # cut EEG to 384 samples
                AF3 = df[0][1:SAMPLES+1].astype(np.float64)
                AF4 = df[1][1:SAMPLES+1].astype(np.float64)
                T7 = df[2][1:SAMPLES+1].astype(np.float64)
                T8 = df[3][1:SAMPLES+1].astype(np.float64)
                Pz = df[4][1:SAMPLES+1].astype(np.float64)

                # assign RAW EEG signals to the makeup of 5 electrode channels
                RAW = np.vstack((AF3, AF4, T7, T8, Pz))
                X[i] = RAW
                del AF3, AF4, T7, T8, Pz, RAW

                # COMMENTED CODE FOR THE FILTERING OF DELTA AND ALPHA SIGNALS 

                # DELTA = np.zeros(RAW.shape)
                # ALPHA = np.zeros(RAW.shape)

                # for c_no in range(RAW.shape[0]):
                    # DELTA[c_no] = mne.filter.filter_data(RAW[c_no], 128, .5, 4, verbose='ERROR').astype(np.float64)
                    # ALPHA[c_no] = mne.filter.filter_data(RAW[c_no], 128, 4, 8, verbose='ERROR').astype(np.float64)

                # ASSIGN DELTA, ALPHA, RAW EEG INTO X
                # X[i] = np.concatenate((DELTA, ALPHA, RAW))
                # X[i] = np.concatenate((DELTA, ALPHA))
                # X[i] = DELTA

                # X[i][1] = np.array([])
                # X[i][2] = np.array([])

                # erase any deleted variables
                gc.collect()

                # assign one-hot-encoded labels into labels array
                y[i][outputstageDict[key]] = 1
                
                
                i += 1
            else:
                # skip any data samples with images that are not RGB or EEG trials that have less than 384 Hz worth of samples  
                print('not 3d and not less than 384', i)            

    print('Amount of samples that meet the 2 criteria:', i)

    # return all data
    return X, X_r, y

# split the data, uncomment this section when saving new sets of data files 

# x_train: EEG
# x_train_r: real images
# y_train: labels
# X_train, X_train_r, y_train = generate_data()

# save new sets of data

# with open('X_train.pkl', 'wb') as f:
#     pickle.dump(X_train, f)
# with open('X_train_r.pkl', 'wb') as f:
#     pickle.dump(X_train_r, f)
# with open('y_train.pkl', 'wb') as f:
#     pickle.dump(y_train, f)

# load in EEG, image, and label data with pickle library
with open('X_train.pkl','rb') as f:
    X_train = pickle.load(f)
with open('X_train_r.pkl','rb') as f:
    X_train_r = pickle.load(f)
with open('y_train.pkl','rb') as f:
    y_train = pickle.load(f)

# a profiler records the total runtime for every function call
profiler = cProfile.Profile()
profiler.enable()

# main creation and training of CVAE-GAN process 
if __name__ == '__main__':
    # create CVAE_GAN class
    model = CVAE_GAN(X_train.shape, y_train.shape)
    # build CVAE-GAN architecture from class object
    net = model.build_model()

    # loop over TRAINING STEP method
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        # train CVAE-GAN
        model.train(X_train[i: i + BATCH_SIZE], X_train_r[i: i + BATCH_SIZE], y_train[i: i + BATCH_SIZE], i)
        print(i)

        # memory issues require the iteration loop be stopped at 250 iterations
        if i == 1000:
            # save generator and CVAE-GAN only for evaluation process
            model.G.save('CVAE-GAN_GEN.h5')
            model.CVAE_GAN.save('CVAE-GAN.h5')

            # disable the profiler when the training is over
            profiler.disable()
            # print the statistics
            stats = pstats.Stats(profiler).sort_stats('tottime')
            stats.print_stats()

