from load_process_images import getImages
from keras.models import load_model
import numpy as np


def autoencode_images():
    images = getImages(return_single=True)

    encoder = load_model('models/encoder_1001.h5')

    return np.asarray(encoder.predict(images))


