'''This script loads images and processes them
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3


def processImage(img, gamma=0.5):
    # original res = 240*320
    image = img[80:230, 80:230]     # crop are of interest
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    # convert from BGR to RGB
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    image = image.astype(float)/255.0     # normalize
    return image


def getImages(return_single=False, VAL=False):
    purple, blue, orange, pu_bl, pu_or, bl_pu, bl_or, or_pu, or_bl, pu_hand, bl_hand, or_hand \
        = [], [], [], [], [], [], [], [], [], [], [], []
    path = '../../data/icub_pointing_1/'
    if VAL:
        path = path + 'validation/'
    for filename in os.listdir(path + 'purple'):
        img = cv2.imread(os.path.join(path + 'purple', filename))
        if img is not None:
            purple.append(processImage(img))
    for filename in os.listdir(path + 'blue'):
        img = cv2.imread(os.path.join(path + 'blue', filename))
        if img is not None:
            blue.append(processImage(img))
    for filename in os.listdir(path + 'orange'):
        img = cv2.imread(os.path.join(path + 'orange', filename))
        if img is not None:
            orange.append(processImage(img))
    for filename in os.listdir(path + 'pu_bl'):
        img = cv2.imread(os.path.join(path + 'pu_bl', filename))
        if img is not None:
            pu_bl.append(processImage(img))
    for filename in os.listdir(path + 'pu_or'):
        img = cv2.imread(os.path.join(path + 'pu_or', filename))
        if img is not None:
            pu_or.append(processImage(img))
    for filename in os.listdir(path + 'bl_pu'):
        img = cv2.imread(os.path.join(path + 'bl_pu', filename))
        if img is not None:
            bl_pu.append(processImage(img))
    for filename in os.listdir(path + 'bl_or'):
        img = cv2.imread(os.path.join(path + 'bl_or', filename))
        if img is not None:
            bl_or.append(processImage(img))
    for filename in os.listdir(path + 'or_pu'):
        img = cv2.imread(os.path.join(path + 'or_pu', filename))
        if img is not None:
            or_pu.append(processImage(img))
    for filename in os.listdir(path + 'or_bl'):
        img = cv2.imread(os.path.join(path + 'or_bl', filename))
        if img is not None:
            or_bl.append(processImage(img))
    for filename in os.listdir(path + 'pu_hand'):
        img = cv2.imread(os.path.join(path + 'pu_hand', filename))
        if img is not None:
            pu_hand.append(processImage(img))
    for filename in os.listdir(path + 'bl_hand'):
        img = cv2.imread(os.path.join(path + 'bl_hand', filename))
        if img is not None:
            bl_hand.append(processImage(img))
    for filename in os.listdir(path + 'or_hand'):
        img = cv2.imread(os.path.join(path + 'or_hand', filename))
        if img is not None:
            or_hand.append(processImage(img))
    if return_single:
        return np.asarray(purple + blue + orange + pu_bl + pu_or + bl_pu + bl_or + or_pu + or_bl + pu_hand + bl_hand + or_hand)
    else:
        return purple, blue, orange, pu_bl, pu_or, bl_pu, bl_or, or_pu, or_bl, pu_hand, bl_hand, or_hand


# testing
#imgs = getImages(return_single=True)
#while True:
    #img = random.choice(imgs)
    #plt.imshow(img)
    #plt.show()
