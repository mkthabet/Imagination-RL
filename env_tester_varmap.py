'''
This script loads an environment model implemented as an MDN and runs it in closed loop after seeding it with an image.
Actions are selected by a pre-trained controller that is also loaded along with an encoder to encode the seeding image
from the environment.
'''

import numpy as np
import random
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from load_process_images import getImages
from mdn import MDN

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 4
NUM_COMPONENTS = 48

def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot

def onehot2int(onehot):
    return np.argmax(onehot)

def mse(x,y):
    return ((x-y)**2).mean()

class PointingEnv:
    def __init__(self, num_items = 3):
        self.num_items = num_items
        self.purple, self.blue, self.orange, self.pu_bl, self.pu_or, self.bl_pu, self.bl_or, self.or_pu, self.or_bl, \
        self.pu_hand, self.bl_hand, self.or_hand = getImages()

        self.env_model = MDN(num_components=NUM_COMPONENTS, in_dim=LATENT_DIM+4, out_dim=LATENT_DIM,
                             model_path="models/env_model_605.h5")
        self.encoder = load_model("models/encoder_1001.h5")
        self.dqn_model = load_model('models/controller_605.h5')
        self.decoder = load_model("models/decoder_1001.h5")
        self.r_model = load_model("models/r_model_605.h5")
        self.encoder_v = load_model("models/encoder_2001.h5")
        self.decoder_v = load_model("models/decoder_2001.h5")

        self.s_bar = None


    def reset(self):
        #self.state is the internal state.
        #self.state = random.randint(0,1) #0 = b, 1 = g, 2 = b_only, 3 = g_only,
        self.state = random.choice([0, 1, 2])
        #self.state = 0
        return self._generateState()


    def step(self, action):
        assert action < self.getActSpaceSize() and action >= 0, "action cannot exceed number of items +1 or be less than 0, action = %r" % action

        done = 0

        if self.state == 0:
            if action == 0:
                self.state = random.choice([5, 7, 9])
                #self.state = 7
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
        elif self.state == 1:
            if action == 1:
                self.state = random.choice([3, 8, 10])
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
        elif self.state == 2:
            if action == 2:
                self.state = random.choice([4, 6, 11])
                #self.state = 11
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
        elif self.state == 3:
            if action == 1:
                self.state = 0
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 4:
            if action == 2:
                self.state = 0
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 5:
            if action == 0:
                self.state = 1
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 6:
            if action == 2:
                self.state = 1
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 7:
            if action == 0:
                self.state = 2
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 8:
            if action == 1:
                self.state = 2
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
        elif self.state == 9 or self.state == 10 or self.state == 11:
            if action == 3:
                done = 1
                reward = 5
                print 'DONE#####################'
            else:
                done = 1
                reward = -1
                print 'mistake.....'

        return self._generateState(), reward, done

    def _generateState(self):
        # 0 = b, 1 = g, 2 = b_only, 3 = g_only, 4 = b_hand, 5 = g_hand
        #print 'state = ' , self.state
        if self.state == 0:
            return random.choice(self.purple)
        elif self.state == 1:
            return random.choice(self.blue)
        elif self.state == 2:
            return random.choice(self.orange)
        elif self.state == 3:
            return random.choice(self.pu_bl)
        elif self.state == 4:
            return random.choice(self.pu_or)
        elif self.state == 5:
            return random.choice(self.bl_pu)
        elif self.state == 6:
            return random.choice(self.bl_or)
        elif self.state == 7:
            return random.choice(self.or_pu)
        elif self.state == 8:
            return random.choice(self.or_bl)
        elif self.state == 9:
            return random.choice(self.pu_hand)
        elif self.state == 10:
            return random.choice(self.bl_hand)
        elif self.state == 11:
            return random.choice(self.or_hand)


    def printState(self):
        state = self._generateState()
        stateArr = np.array(state)
        print "state = %s" % stateArr.T

    def getStateSpaceSize(self):
        return ( IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def getActSpaceSize(self):
        return self.num_items+1

    def encode(self, s):
        encoded = self.encoder.predict(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
        encoded = np.asarray(self.encoder_v.predict(encoded))
        return encoded[0, 0, :]

    def model_reset(self, s_zero):
        print('resetting model...')
        self.s_bar = self.encode(s_zero)
        return self.s_bar

    def model_step(self, a):
        #print self.s_bar.shape
        s_a = np.append(self.s_bar, int2onehot(a,self.getActSpaceSize()))
        #print s_a.shape
        mu, sigma, pi = self.env_model.get_dist_params(s_a.reshape(1, -1))
        #print('model out = ', model_out)
        #print('coefficients = ', pi)
        component = np.random.choice(np.arange(0, NUM_COMPONENTS, 1), p=pi.flatten())
        mu = mu[0, component, :]
        #z = np.random.normal(mu, np.exp(sigma[0, component, :]))
        # z = np.zeros([LATENT_DIM,])
        # for i in range(NUM_COMPONENTS):
        #     z_log_var = z_log_vars[:, i * LATENT_DIM:(i + 1) * LATENT_DIM]
        #     component = np.random.normal(loc=z_means[:, i], scale=np.exp(z_log_vars[:, i]/2))
        #     z = z + component*coefficients[i]
        #self.s_bar = self.s_bar + z
        self.s_bar = mu
        #self.s_bar = model_out
        r_out = self.r_model.predict(s_a.reshape((1,s_a.size)))
        r = r_out[0].flatten()
        done = r_out[1].flatten()
        #print('means = ', mu)
        #print('sigmas = ', sigma)

        return self.s_bar, r, done

    def act(self, s):
        out = self.dqn_model.predict(np.reshape(s, (1,LATENT_DIM)))
        return np.argmax(out)


testEnv = PointingEnv()
s = testEnv.reset()
s_hat = testEnv.model_reset(s)
d = d_hat = 0
episodes = 0
MAX_EPISODES = 50
log = []
misclass_r = 0
misclass_d = 0
im_size = 64
figure = np.zeros((im_size, im_size, 3))
while(episodes < MAX_EPISODES):
    if round(d_hat) == 1:
        #print 'new episode'
        episodes = episodes + 1
        d_hat = 0
        s = testEnv.reset()
        s_hat = testEnv.model_reset(s)
    figure[:, 0:im_size, :] = s
    decoded = testEnv.decoder_v.predict(s_hat.reshape(1, LATENT_DIM))
    decoded = testEnv.decoder.predict(decoded.reshape(1, LATENT_DIM))
    figure[:, 0:im_size, :] = decoded
    #print 'mse(s): ', mse(testEnv.encode(s), s_hat)#, ', r: ', r, ', r_hat', r_hat, ', d: ', d, ', d_hat', d_hat
    #print('s_hat', s_hat)

    #a = testEnv.act(testEnv.encode(s))
    a = testEnv.act(s_hat)
    print ('z = ', s_hat, 'a = ', a)
    plt.figure()
    plt.imshow(figure)
    plt.axis('off')
    plt.show()
    #s, r, d = testEnv.step(a)
    #s_hat, r_hat, d_hat = testEnv.model_step(a)
    s_hat, r_hat, d_hat = testEnv.model_step(a)
    #print ('action = ', a)

    #print 'r: ', r, ', r_hat', r_hat, ', d: ', d, ', d_hat', d_hat
    print 'r_hat: ', r_hat, 'd_hat: ', d_hat
    #print 'mse:  s: ', mse(testEnv.get_sbar(s),s_hat), ', r: ' , mse(r,r_hat) , ', d: ' , mse(d,d_hat)

    #if (round(r)-round(r_hat)) != 0:
        #misclass_r += 1
    #if (round(d)-round(d_hat)) != 0:
        #misclass_d += 1
    #log.append([mse(testEnv.encode(s), s_hat), mse(r, r_hat), mse(d, d_hat)])
#print 'misclass(r) = ', misclass_r, ' , misclass(d) = ', misclass_d






