'''
CVAE-GAN: MODEL ARCHITECTURE AND TRAINING STEP

CALLED IN TRAINING.py
'''

# IMPORTS

# ML LIBRARIES
from keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Layer, InputSpec
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Conv2DTranspose, LSTM, CuDNNLSTM
from keras.layers import Lambda, Input, Flatten, Dense, Activation, Concatenate, RepeatVector, TimeDistributed, Reshape, Embedding, Dropout, multiply
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from keras.models import Model, Sequential
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
from tensorflow.keras.utils import plot_model

# STD LIBRARIES
import os
import gc
import math

# FILE IMPORTS
from CONSTANTS import *


class CVAE_GAN:
    def __init__(self, EEG_SHAPE, CATEGORY_SHAPE):
        '''
        Instantiates specific class attributes: model variables and data shapes
        '''
        # model variables
        self.E = None
        self.D = None   
        self.G = None
        self.C = None
        self.CVAE_GAN = None
    
        # shape constants
        self.EEG_SHAPE = EEG_SHAPE
        self.IMG_INPUT_SHAPE = IMG_SHAPE
        self.NOISE_SIZE = CATEGORY_SHAPE[-1]
        self.c = CATEGORY_SHAPE

    def train(self, x_batch, x_r_batch, y_batch, idx):
        '''
        Training step for CVAE-GAN; data has batch sizes of 4

        x_batch: EEG data batch from TRAINING file
        x_r_batch: image data batch from TRAINING file
        y_batch: categories for each image/EEG from TRAINING file 
        idx: corresponds to i iterations in main training loop
        
        precondition: all data is obtained with corresponding labels of the same first dimension
        postcondition: returns losses of discriminator and encoder networks
        '''

        c_flatten = np.argmax(y_batch, axis=1)
        c = c_flatten.reshape(-1, 1)

        # classifier labels
        c_dummy = np.ones(y_batch.shape, dtype='float32')
        # discriminator labels for both real and fake images
        y_dummy = np.concatenate([np.full(BATCH_SIZE, .9), np.full(BATCH_SIZE, .1)])
        y_dummy += .05 * np.random.random(y_dummy.shape)

        # train encoder on batch in order to get real labels into the network
        e_loss = self.E.train_on_batch([x_batch, c], y_batch)

        # obtain EEG latent vectors to be put into the generator 
        latent = self.E.predict([x_batch, c])

        # may need to concatenate noise in order to randomize input
        # gen_input = np.concatenate([latent, np.random.randn(BATCH_SIZE, self.NOISE_SIZE)])
        
        # obtain generator image with latent vector and conditional label as input
        enc_img = self.G.predict([latent, c])

        # display/save generated image and real images into a file
        self.generate_samples(x_r_batch, enc_img, idx)

        # train discriminator successively with real images and "fake" generator images
        dis_x = np.concatenate([x_r_batch, enc_img])
        d_loss = self.D.train_on_batch(dis_x, y_dummy) # d_acc

        # generator training is optional, as CVAE-GAN network is wholly trained
        # g_loss = self.G.train_on_batch([latent, np.random.randint(0, self.c[-1], BATCH_SIZE)], np.ones(BATCH_SIZE))

        # classifier trained with real images and classifier labels of Numpy Ones array
        self.C.train_on_batch(x_r_batch, c_dummy)

        # train CVAE-GAN model with EEG signals and conditional input with Numpy Ones as the "real" labels
        self.CVAE_GAN.train_on_batch([x_batch, c], np.ones(BATCH_SIZE))

        # ensure any variables deleted during the training process are erased from memory
        gc.collect()
        K.clear_session()

        # return loss for statistical methods
        loss = {
            'e_loss': e_loss,
            'd_loss': d_loss,
        }

        return loss
    
    def generate_samples(self, r_img, img, idx):
        '''
        Displays real and generator's images and saves the plot into a file

        r_img: real image from current training iteration
        img: image produced by the generator
        idx: index of the training loop solely used to number saved images

        precondition: two sets of images are obtained
        postcondition: saves figure of batch size 4
        '''
        DS = int(math.sqrt(BATCH_SIZE))

        count = 0
        fig, axes = plt.subplots(DS ** 2, DS)
        fig.set_size_inches(DS ** 2, DS)

        for i in range(DS ** 2):
            axes[i, 0].imshow(cv2.resize(img[count], (IMG_SHAPE[0], IMG_SHAPE[1])))
            axes[i, 1].imshow(cv2.resize(r_img[count], (IMG_SHAPE[0], IMG_SHAPE[1])))

            count += 1

        # save figure
        plt.savefig(r'C:\Users\bluet\Desktop\MemoryReconstruction\generated_real\generated_images_{0}.png'.format(idx // BATCH_SIZE))
        # plt.show()

        # clear figure for just generator images
        plt.clf()

        count = 0
        for i in range(idx, idx + BATCH_SIZE):             
            plt.imshow(cv2.resize(img[count], (IMG_SHAPE[0], IMG_SHAPE[1])))

            plt.axis('off')
            plt.savefig(r'C:\Users\bluet\Desktop\MemoryReconstruction\predictions\generated{0}.png'.format(i), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.show()

            count += 1

    def build_model(self):
        '''
        Constructs CVAE-GAN framework and saves other models into class attributes

        x_r: real image input
        x_c: conditional label input

        return: CVAE-GAN framework
        '''
        # CONSTRUCT MODELS

        self.E = self.build_encoder()
        self.D = self.build_discriminator()   
        self.G = self.build_generator()
        self.C = self.build_classifier()

        # INPUTS
        x_e = Input(shape=(self.EEG_SHAPE[-2], self.EEG_SHAPE[-1]))
        x_c = Input(shape=1)

        # ENC
        E_out = self.E([x_e, x_c])

        # check equal shapes, can't do this in tf==2.7.0
        # assert E_out.shape == self.G.layers[0].output_shape[0]

        # GEN
        x_f = self.G([E_out, x_c])

        # DISCRIM
        y_r = self.D(x_f)

        # CLASSIFIER
        c_f = self.C(x_f)

        # BUILD MODELS & COMPILE
        self.C.compile(loss='mse', optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.C.summary()

        self.G.summary()

        self.D.compile(loss='mse', optimizer=Adam(lr=2.0e-4, beta_1=.5), metrics=['accuracy'])
        self.D.summary()

        self.E.compile(loss='binary_crossentropy', optimizer='adam')
        self.E.summary()

        self.D.trainable = False

        self.CVAE_GAN = Model([x_e, x_c], [y_r, c_f])
        self.CVAE_GAN.compile(loss='mse', optimizer=Adam(lr=9.0e-4, beta_1=0.5))
        plot_model(self.CVAE_GAN)
        self.CVAE_GAN.summary()

        return self.CVAE_GAN

    def build_encoder(self):
        '''
        Build encoder architecture: requires EEG data and conditional inputs along with image category labels
        '''
        x0 = Input(shape=(self.EEG_SHAPE[-2], self.EEG_SHAPE[-1]))

        x = LSTM(100)(x0)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        
        x = Flatten()(x)

        c_inputs = Input(shape=1)
        x = Concatenate(axis=-1)([x, c_inputs])

        x = Dense(self.c[-1], activation='sigmoid')(x)

        net = Model([x0, c_inputs], x)

        net.summary()

        return net


    def build_discriminator(self):
        '''
        Builds discriminator architecture with downsampling blocks (convolutional neural network)
        '''
        x0 = Input(shape=self.IMG_INPUT_SHAPE)

        layers = [64, 128, 256, 128, 64]
        
        x = self.ds_blocks(x0, layers[0])

        for l in layers[1:]:
            x = self.ds_blocks(x, l)

        x = GlobalAveragePooling2D()(x)
        x = LeakyReLU(0.3)(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        dis = Model(x0, x)

        dis.summary()

        return dis


    def build_generator(self):
        '''
        Builds upsampling generator architecture with two inputs: latent vector from encoder and conditional input
        '''
        z_inputs = Input(shape=self.c[-1])
        c_inputs = Input(shape=1)

        # multiply inputs as Concatenate() function may have mismatching shapes
        z = multiply([z_inputs, c_inputs])

        w = self.IMG_INPUT_SHAPE[0] // (2 ** 4)

        x = Dense(w * w * 2)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 2))(x)

        x = Conv2DTranspose(filters=256, strides=2, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(filters=128, strides=2, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(filters=64, strides=2, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(filters=32, strides=2, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        d = self.IMG_INPUT_SHAPE[-1]
        x = Conv2DTranspose(filters=d, strides=1, kernel_size=3, padding='same', activation='tanh')(x)

        dec = Model([z_inputs, c_inputs], x)

        dec.summary()

        return dec

    def build_classifier(self):
        '''
        Builds classifier architecture (convolutional neural network)
        '''
        inputs = Input(shape=self.IMG_INPUT_SHAPE)

        x = Conv2D(filters=64, strides=2, kernel_size=3)(inputs)

        for k in [128, 256, 128, 64]:
            x = Conv2D(k, strides=2, kernel_size=3)(x)

        x = Flatten()(x)

        x = Dense(self.c[1])(x)
        x = Activation('sigmoid')(x)

        clf = Model(inputs, x)

        clf.summary()

        return clf

    def ds_blocks(self, x, k):
        '''
        Lays out framework for downsampling blocks for convolutional neural network (discriminator)
        '''
        const = max_norm(0.1)

        x = Conv2D(k, kernel_size=3, padding='same', kernel_constraint=const)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=.3)(x)

        return x
