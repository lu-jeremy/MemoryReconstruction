'''
CVAE-GAN PERFORMANCE EVLAUATION
'''

# IMPORTS

# ML LIBRARIES
from keras.models import load_model
from keras_flops import get_flops
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# FILE IMPORT
from CONSTANTS import *

# INCEPTION SCORE IMPORTS
from math import floor
import numpy as np
from numpy import expand_dims, log, mean, std, exp, asarray
from numpy.random import shuffle
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# load models in
generator = load_model('CVAE-GAN_GEN.h5')
CVAE_GAN = load_model('CVAE-GAN.h5')

# IMAGE INCEPTION SCORE
def calculate_inception_score(images, n_split=10, eps=1E-16):
    '''
    Calculate Inception Score from a set of images
    '''
    # load inception v3 model
    model = InceptionV3()
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype('float32')
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = mean(sum_kl_d)
        is_score = exp(avg_kl_d)
        scores.append(is_score)

    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std
 
# the generator will produce 1000 images
AMT_FILES = 1048

def retrieve_images():
    '''
    Retrieves generator images from prediction folder

    return: image set resized to 299x299
    '''
    images = np.ones((AMT_FILES, 299, 299, IMG_SHAPE[2]))
    predictions_dir = r'C:\Users\bluet\Desktop\MemoryReconstruction\predictions'

    # walks through predictions folder
    for root, dir, files in os.walk(predictions_dir):
        # goes through all images
        for i, filename in enumerate(files):
            # read and resize images to 128x128
            img = plt.imread(os.path.join(root, filename))
            # eliminate transparency dimension
            img = np.delete(img, -1, axis=2)
            img = cv2.resize(img, (299, 299))
            # store the image in the overall images array
            images[i] = img

    # randomize order of images and return them
    shuffle(images)
    return images

# call retrieve_images function
predicted_images = retrieve_images()

# calculate inception score
avg, std = calculate_inception_score(predicted_images)
# IS Average: 1.0021455
print('IS Average:', avg)
# IS Standard Deviation: 0.00018228502
print('IS Standard Deviation:', std)

# MODEL FLOATING OPERATIONS
# 4.09 GIGAFLOPS
gen_flops = get_flops(generator, batch_size=BATCH_SIZE)
print(f"FLOPS: {gen_flops / 10 ** 9:.03} G")

# 102 GIGAFLOPS
model_flops = get_flops(CVAE_GAN, batch_size=BATCH_SIZE)
print(f"FLOPS: {model_flops / 10 ** 9:.03} G")

