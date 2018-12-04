'''
This script loads a pre-trained controller and uses it to generate rollouts from an environment.
It is used to test the controller.
'''
from __future__ import division

import random, numpy, math, gym

#-------------------- BRAIN ---------------------------

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from pointing_env import PointingEnv

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 8

VAE_VER = '0008_1'
MODEL_VER = '0001'

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.controller, self.encoder = self._createModel()

    def _createModel(self):
        controller = load_model('models/controller_' + MODEL_VER + ".h5")
        encoder = load_model('models/encoder_' + VAE_VER + ".h5")

        return controller, encoder

    def predict(self, s):
            return self.controller.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, LATENT_DIM)).flatten()

    def encode(self, s):
        encoded = np.asarray(self.encoder.predict(s))
        return encoded[ 0, :]

BATCH_SIZE = 64


class Agent:
    steps = 0

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)

    def act(self, s):
        return numpy.argmax(self.brain.predictOne(s))


#-------------------- ENVIRONMENT ---------------------
done_cnt = 0
failed_cnt = 0
R_total = 0
class Environment:
    def __init__(self, num_items, use_all=False, val=False):
        self.env = PointingEnv(num_items=num_items, use_all=use_all, val=val)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0
        global failed_cnt
        global done_cnt
        global R_total
        while True:
            sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            a = agent.act(sbar)
            #print a
            s_, r, done = self.env.step(a)

            s = s_
            R += r

            if done:
                if r < 1:
                    failed_cnt += 1
                else:
                    done_cnt += 1
                    R_total += R
                break

        #print("Total reward:", R)

#-------------------- MAIN ----------------------------

def test_model(vae_ver, model_ver, use_all=False, val=True):
    global VAE_VER
    global MODEL_VER
    VAE_VER = vae_ver
    MODEL_VER = model_ver

    num_items = 3
    env = Environment(num_items=num_items, use_all=use_all, val=val)

    stateCnt = env.env.getStateSpaceSize()
    actionCnt = env.env.getActSpaceSize()

    agent = Agent(stateCnt, actionCnt)

    episodes = 0
    runs = 0
    MAX_EPISODES = 1000
    MAX_RUNS = 1
    total_done_cnt = 0
    global R_total
    R_total = 0
    while runs < MAX_RUNS:
        while episodes < MAX_EPISODES:
            env.run(agent)
            episodes += 1
        # agent.brain.model.save("point_3.h5")
        runs += 1
        global done_cnt
        total_done_cnt += done_cnt
        done_cnt = 0
        episodes = 0
    avg_done_cnt = total_done_cnt*100 / (MAX_RUNS*MAX_EPISODES)
    avg_R = R_total / (total_done_cnt)
    print('avg done: ', avg_done_cnt, '. avg R: ', avg_R)
    return avg_done_cnt, avg_R
    # env.run(agent, False)
    #print('Average done count: ', avg_done_cnt)
    #print('Average R: ', R_total / (MAX_RUNS * MAX_EPISODES))

#test_model()