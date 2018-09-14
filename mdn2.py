import random, math, gym
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, Lambda, Concatenate
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from keras import backend as K
from keras import metrics
# from pointing_model import PointingEnv
import matplotlib.pyplot as plt

IN_DIM = 1
OUT_DIM = 1
NUM_COMPONENTS = 24
BATCH_SIZE = None


class MDN:
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM, num_components=NUM_COMPONENTS):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_components = num_components
        self.model, self.model_train = self._createModel()

    def _createModel(self):
        def get_mixture_coef(output, numComonents=24, outputDim=1):
            out_pi = output[:, :numComonents]
            out_sigma = output[:, numComonents:2 * numComonents]
            out_mu = output[:, 2 * numComonents:]
            out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
            out_mu = K.permute_dimensions(out_mu, [1, 0, 2])
            # use softmax to normalize pi into prob distribution
            #max_pi = K.max(out_pi, axis=1, keepdims=True)
            #out_pi = out_pi - max_pi
            #out_pi = K.exp(out_pi)
            #normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
            #out_pi = normalize_pi * out_pi
            # use exponential to make sure sigma is positive
            out_sigma = K.exp(out_sigma)
            return out_pi, out_sigma, out_mu

        def tf_normal(y, mu, sigma):
            oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
            result = y - mu
            result = K.permute_dimensions(result, [2, 1, 0])
            result = result * (1 / (sigma + 1e-8))
            result = -K.square(result) / 2
            result = K.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
            result = K.prod(result, axis=[0])
            return result

        def get_lossfunc(out_pi, out_sigma, out_mu, y):
            result = tf_normal(y, out_mu, out_sigma)
            result = result * out_pi
            result = K.sum(result, axis=1, keepdims=True)
            result = -K.log(result + 1e-8)
            return K.mean(result)

        def mdn_loss(numComponents=24, outputDim=1):
            def loss(y, output):
                out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
                return get_lossfunc(out_pi, out_sigma, out_mu, y)

            return loss

        model_input = Input(shape=(1,), name='model_in')
        model_out = Dense(units=24, activation='tanh', name='model_dense4')(model_input)
        # model_out = Dense(units=2 * (1 * 24) + 24, name='model_out')(model_out)
        out_pi = Dense(units=NUM_COMPONENTS, activation='softmax', name='out_pi')(
            model_out)
        out_mu = Dense(units=OUT_DIM * NUM_COMPONENTS, name='out_mu')(model_out)
        out_sigma = Dense(units=OUT_DIM * NUM_COMPONENTS, name='out_sigma')(model_out)
        out_concat = Concatenate()([out_pi, out_sigma, out_mu])
        model = Model(inputs=model_input, outputs=out_concat)
        opt = Adam(lr=0.001)
        model.compile(loss=mdn_loss(), optimizer=opt)
        # model_train.summary()
        return model, model

    def train_model(self, x, y, epoch=10000, verbose=1):
        self.model_train.fit(x, y, batch_size=2500, epochs=epoch, verbose=verbose)

    def get_dist_params(self, x):
        model_out = np.asarray(self.model_train.predict(x))
        #means = model_out[:, 0:self.out_dim * self.num_components]
        #log_vars = model_out[:, self.out_dim * self.num_components:self.out_dim * self.num_components * 2]
        #coeffs = model_out[:, self.out_dim * self.num_components * 2:]
        out_pi = model_out[:, :NUM_COMPONENTS]
        out_sigma = model_out[:, NUM_COMPONENTS:2 * NUM_COMPONENTS]
        out_mu = model_out[:, 2 * NUM_COMPONENTS:]
        out_mu = np.reshape(out_mu, [-1, NUM_COMPONENTS, OUT_DIM])
        out_mu = np.transpose(out_mu, [1, 0, 2])
        # use softmax to normalize pi into prob distribution
        #max_pi = np.amax(out_pi, 1, keepdims=True)
        #out_pi = out_pi - max_pi
        #out_pi = np.exp(out_pi)
        #normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
        #out_pi = normalize_pi * out_pi
        #print 'out shape:', model_out.shape
        # means = model_out[0, :, :]
        #print 'means shape:', means.shape
        # log_vars = model_out[1, :, :]
        #print 'vars shape:', log_vars.shape
        # coefficients = model_out[2, :, :]
        #print 'coeffs shape:', coeffs.shape
        return out_mu, out_sigma, out_pi


####### MAIN #########

NSAMPLE = 250

y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))  # random noise
x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)

# plt.figure(figsize=(8, 8))
# plt.plot(x_data,y_data,'ro', alpha=0.3)
# plt.show()

mdn = MDN()
mdn.train_model(x_data, y_data)
means, sigmas, coeffs = mdn.get_dist_params(x_data)

x_test = np.float32(np.arange(-15, 15, 0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST, 1)  # needs to be a matrix, not a vector


def generate_ensemble(out_mu, out_sigma, out_pi, M=1):
    result = np.random.rand(NTEST, M, OUT_DIM)
    rn = np.random.randn(NTEST, M)
    mu = 0
    std = 0
    idx = 0

    # transforms result into random ensembles
    for j in range(0, M):
        for i in range(0, NTEST):
            for d in range(0, OUT_DIM):
                idx = np.random.choice(24, 1, p=out_pi[i])
                mu = out_mu[idx, i, d]
                std = out_sigma[i, idx]
                result[i, j, d] = mu + rn[i, j] * std
    return result


y_test = generate_ensemble(means, sigmas, coeffs)

x_test = np.float32(np.arange(-15.0, 15.0, 0.1))
x_test = x_test.reshape(x_test.size, 1)

plt.figure(figsize=(8, 8))
plt.plot(x_data, y_data, 'ro', x_test, y_test[:, :, 0], 'bo', alpha=0.3)
plt.show()
