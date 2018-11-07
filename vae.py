
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.stats import norm
import random

from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from load_process_images import getImages

batch_size = 200
latent_dim = 8
epochs = 200
epsilon_std = 1.0
BETA = 32
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
original_dim = IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS
BETA_NORM = BETA*latent_dim/original_dim

def vae_loss(x, x_decoded_mean):
    x= K.batch_flatten(x)
    x_decoded_mean = K.batch_flatten(x_decoded_mean)
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    #xent_loss = original_dim*metrics.mean_squared_error(x, x_decoded_mean)
    #xent_loss = K.sum(K.square(x_decoddded_mean-x), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + BETA*kl_loss
    #return kl_loss+1


img_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', data_format='channels_last')(img_input)
conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
conv = Conv2D(128, (4, 4), strides=(2, 2), activation='relu')(conv)
conv2layer = Conv2D(256, (4, 4), activation='relu')
conv_out = conv2layer(conv)
conv_out_layer = Flatten(name='flatten')
h = conv_out_layer(conv_out)
z_mean = Dense(units=latent_dim)(h)
z_log_var = Dense(units=latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

print(conv2layer.output_shape[1:])


decoder_input = Input(shape=(latent_dim,))
h_decoded = Dense(units=1024, activation='relu')(decoder_input)
h_decoded = Reshape((1, 1, 1024))(h_decoded)
deconv = Conv2DTranspose(128, (5, 5), strides= (2,2), activation='relu')(h_decoded)
deconv = Conv2DTranspose(64, (5, 5), strides= (2,2), activation='relu')(deconv)
deconv = Conv2DTranspose(32, (6, 6), strides= (2,2), activation='relu')(deconv)
decoded_mean = Conv2DTranspose(3, (6, 6), strides= (2,2), activation='sigmoid')(deconv)

encoder = Model(img_input, [z_mean, z_log_var], name='encoder')
decoder = Model(decoder_input, decoded_mean, name='decoder')
decoder.summary()
reconstructed = decoder(z)
vae = Model(img_input, reconstructed, name='vae')
opt = RMSprop(lr=0.00025)
vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

x_train = y_train = x_test = y_test = getImages(return_single=True, use_all=False, val=False)


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

n = 15
digit_size = 64
figure = np.zeros((digit_size * n, digit_size * n, 3))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        #z_sample = np.array([[xi, yi, xi, yi]])
        z_sample = np.random.normal(0, 1, (1, latent_dim))
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size, :] = digit

plt.figure()
plt.imshow(figure)
plt.show()

plt.plot(history.history['loss'])
plt.show()