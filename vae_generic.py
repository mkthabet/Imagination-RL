
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.stats import norm
import random

from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
from keras import metrics
from autoencode_images import *
from load_process_images import getImages

batch_size = 600
latent_dim = 4
epochs = 500
epsilon_std = 1.0
BETA = 16
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
original_dim = 4
BETA_NORM = BETA*latent_dim/original_dim

def vae_loss(x, x_decoded_mean):
    x= K.batch_flatten(x)
    x_decoded_mean = K.batch_flatten(x_decoded_mean)
    xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
    #xent_loss = original_dim*metrics.mean_squared_error(x, x_decoded_mean)
    #xent_loss = K.sum(K.square(x_decoddded_mean-x), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + BETA*kl_loss
    #return kl_loss+1


input = Input(shape=(original_dim,))
h = BatchNormalization()(input)
h = Dense(units=256, activation='relu')(h)
#h = BatchNormalization()(h)
#h = Activation('relu')(h)
h = Dense(units=128, activation='relu')(h)
h = Dense(units=64, activation='relu')(h)
z_mean = Dense(units=latent_dim)(h)
z_log_var = Dense(units=latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


decoder_input = Input(shape=(latent_dim,))
h_decoded = Dense(units=256, activation='relu')(decoder_input)
h_decoded = Dense(units=128, activation='relu')(h_decoded)
h_decoded = Dense(units=64, activation='relu')(h_decoded)
decoded_mean = Dense(units=original_dim, activation='relu')(h_decoded)


encoder = Model(input, [z_mean, z_log_var], name='encoder')
decoder = Model(decoder_input, decoded_mean, name='decoder')
decoder.summary()
reconstructed = decoder(z)
vae = Model(input, reconstructed, name='vae')
opt = RMSprop(lr=0.00025)
vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

x_train = y_train = x_test = y_test = autoencode_images()


try:
    history = vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))
finally:
    sss = 0
    encoder.save('models/encoder_2001.h5')
    decoder.save('models/decoder_2001.h5')
    vae.save('models/vae_2001.h5')

plt.plot(history.history['loss'])
plt.show()