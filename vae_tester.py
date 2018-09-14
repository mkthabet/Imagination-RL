'''
This script tests variational autoencoder models and shows traversal plots of encoded images.
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from sklearn.manifold import TSNE

from keras.models import load_model
from load_process_images import getImages

latent_dim = 16
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3

decoder = load_model('models/decoder_206.h5')
encoder = load_model('models/encoder_206.h5')

imgs_list = []
for i in getImages():
    imgs_list.append(i)

imgs = getImages(True)

im_size = 64

while True:
    c = random.choice(range(0, imgs.shape[0]))
    img = imgs[c, :, :]
    print(c)
    #img = imgs[117]
    img = img.reshape((1, 64, 64, 3))
    encoded = np.asarray(encoder.predict(img))
    encoded_logvar = encoded[1, :, :]    #store log(var) vector for later
    encoded = encoded[0, :, :]   #get just means
    print(encoded.shape)
    print('means = ', encoded)
    print('std. dev = ', np.sqrt(np.exp(encoded_logvar)))
    decoded = decoder.predict(encoded)
    n = 10
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    figure = np.zeros((im_size*latent_dim, im_size * (n + 2), 3))
    for i in range(latent_dim):
        figure[im_size*i:im_size*(i+1), 0:im_size, :] = img
        figure[im_size*i:im_size*(i+1), im_size:im_size*2, :] = decoded
        for j, xj in enumerate(grid_x):
            #print(encoded[0, i])
            encoded_=np.copy(encoded)
            encoded_[0,i] = xj
            decoded_ = decoder.predict(encoded_)
            figure[im_size*i:im_size*(i+1), im_size*(j+2):im_size*(j+3), :] = decoded_
    plt.figure()
    plt.imshow(figure)
    plt.show()