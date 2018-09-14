import random, math, gym
import numpy as np

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout
from keras.optimizers import *
from keras.models import Model, load_model, model_from_json
from pointing_env import PointingEnv
import matplotlib.pyplot as plt
from mdn import MDN

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 4

ENV_LEARN_START = 0   #number of episodes before training env model starts`
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1 #0.8
MIN_EPSILON = 0.001 #0.0001
LAMBDA = 0.001      # speed of decay+
MAX_EPISODES = 1000
USE_TARGET = False
UPDATE_TARGET_FREQUENCY = 5
NUM_COMPONENTS = 48

epsilon_std = 1.0
BETA = 0.0
episodes = 0
ACTION_CNT = 4

def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot

class EnvironmentModel:
    def __init__(self):
        self.s_bar = None
        self.env_model = MDN(num_components=NUM_COMPONENTS, in_dim=LATENT_DIM + 4, out_dim=LATENT_DIM,
                             model_path="models/env_model_401.h5")
        self.r_model = load_model("models/r_model_401.h5")
        self.decoder = load_model("models/decoder_105.h5")
        self.statecnt = LATENT_DIM

    def init_model(self, s_bar):
        self.s_bar = s_bar
        print('new episode ')
        #plt.imshow(decoded.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        #plt.show()


    def step(self, a):
        s_a = np.append(self.s_bar, int2onehot(a, ACTION_CNT))
        mu, sigma, pi = self.env_model.get_dist_params(s_a.reshape(1, -1))
        component = np.random.choice(np.arange(0, NUM_COMPONENTS, 1), p=pi.flatten())
        mu = mu[0, component, :]
        self.s_bar = mu
        # self.s_bar = model_out
        r_out = self.r_model.predict(s_a.reshape((1, s_a.size)))
        r = r_out[0].flatten()
        done = r_out[1].flatten()

        return self.s_bar, r, done

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.controller, self.encoder, self.controller_target = self._createModel()

    def _createModel(self):
        encoder = load_model('models/encoder_105.h5')

        controller_input = Input(shape=(LATENT_DIM,), name='controller_input')
        controller_out = Dense(units=512, activation='relu')(controller_input)
        controller_out = Dense(units=256, activation='relu')(controller_out)
        #controller_out = Dense(units=32, activation='relu')(controller_out)
        #controller_out = Dense(units=16, activation='relu')(controller_out)
        controller_out = Dense(units=actionCnt, activation='linear')(controller_out)
        controller = Model(inputs=controller_input, outputs=controller_out)
        controller_opt = adam(lr=0.00025)
        controller.compile(loss='mse', optimizer='adam')
        controller.summary()

        # just copy the architecture
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        return controller, encoder, controller_target

    def train(self, x, y, epoch=1, verbose=0):
        self.controller.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.controller_target.predict(s)
        else:
            return self.controller.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, LATENT_DIM), target).flatten()

    def encode(self, s):
        p_z =  np.asarray(self.encoder.predict(s))
        return p_z[0, 0, :]

    def updateTargetModel(self):
        self.controller_target.set_weights(self.controller.get_weights())
        

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ , d)
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        self.imaginary_memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample, imaginary=False):  # in (s, a, r, s_, done) format
        if not imaginary:
            self.memory.add(sample)
        else:
            self.imaginary_memory.add(sample)

        if USE_TARGET and (self.steps % UPDATE_TARGET_FREQUENCY == 0):
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self, imaginary=False):
        if not imaginary:
            batch = self.memory.sample(BATCH_SIZE)
        else:
            batch = self.imaginary_memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        no_state = np.zeros(LATENT_DIM)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=USE_TARGET)

        x = np.zeros((len(batch), LATENT_DIM))
        y = np.zeros((len(batch), self.actionCnt))

        x_env = np.zeros((len(batch), LATENT_DIM + actionCnt))
        y_env_s = np.zeros((len(batch), LATENT_DIM))
        y_env_r = np.zeros((len(batch), 1))
        y_env_d = np.zeros((len(batch), 1))

        for i in range(batchLen):
            o = batch[i]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3];
            done = o[4]

            t = p[i]
            # print (t)
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t
            x_env[i] = np.append(states[i], int2onehot(a, actionCnt))
            y_env_s[i] = states_[i]
            y_env_r[i] = r
            y_env_d[i] = done

        self.brain.train(x, y)

        #if episodes > ENV_LEARN_START:
            #self.brain.train_env(x_env, y_env_s)
            #self.brain.train_r(x_env, [y_env_r, y_env_d])

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)
        self.env_model = EnvironmentModel()

    def run(self, agent):
        s = self.env.reset()
        sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
        R = 0
        imaginary = False
        if episodes>0:
            imaginary = True
            self.env_model.init_model(sbar)
        #TODO: decide if imaginary or not
        while True:
            a = agent.act(sbar)
            #print('a = ', a)
            if imaginary:
                sbar_, r, done = self.env_model.step(a)
                r, done = round(r), round(done)
            else:
                s_, r, done = self.env.step(a)
                sbar_ = agent.brain.encode(np.reshape(s_, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))

            if done: # terminal state
                sbar_ = None

            agent.observe((sbar, a, r, sbar_,done), imaginary=imaginary)
            agent.replay(imaginary=imaginary)

            if imaginary:
                sbar = sbar_
            else:
                s = s_
                sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            R += r

            if done:
                break

        print("Total reward:", R, ", episode: ", episodes)

#-------------------- MAIN ----------------------------
num_items = 3;
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

episodes = 0

try:
    while episodes < MAX_EPISODES:
        env.run(agent)
        episodes = episodes + 1
finally:
    ss=0    #blah blah
    #agent.brain.model.save("models/model_26.h5")
    #agent.brain.env_model.save("models/env_model_26.h5")
    agent.brain.controller.save('models/controller_1001.h5')
    #agent.brain.conv_model.save("models/conv_model_26.h5")
#env.run(agent, False)
