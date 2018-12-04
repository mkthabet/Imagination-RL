import random, math
import numpy as np
from keras.layers import Input, Dense
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from pointing_env import PointingEnv
import matplotlib.pyplot as plt
from mdn import MDN
from sumtree import SumTree

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 4

ENV_LEARN_START = 0   #number of episodes before training env model starts`
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 0.6 #0.8
MIN_EPSILON = 0.0001 #0.0001
LAMBDA = 0.01      # speed of decay+
MAX_EPISODES = 300
USE_TARGET = False
UPDATE_TARGET_FREQUENCY = 5
NUM_COMPONENTS = 48

epsilon_std = 1.0
BETA = 0.0
episodes = 0

def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot

def onehot2int(onehot):
    return np.argmax(onehot)

def mse(x,y):
    return ((x-y)**2).mean()

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.controller, self.encoder, self.controller_target, self.r_model = self._createModel()
        self. env_model = MDN(num_components=NUM_COMPONENTS, in_dim=LATENT_DIM+self.actionCnt, out_dim=LATENT_DIM)

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

        #just copy the architecure
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        r_model_input = Input(shape=(LATENT_DIM+actionCnt,), name = 'r_in')
        r_model_out = Dense(units=512, activation='relu', name = 'r_dense1')(r_model_input)
        r_model_out = Dense(units=256, activation='relu', name = 'r_dense2')(r_model_out)
        r_out = Dense(units=1, name='r_out', activation='linear')(r_model_out)
        d_out = Dense(units = 1, activation= 'sigmoid', name = 'd_out')(r_model_out)
        r_model = Model(r_model_input, [r_out, d_out])
        r_model.compile(loss='mse', optimizer='adam')

        return controller, encoder, controller_target, r_model

    def train_controller(self, x, y, epoch=1, verbose=0):

        self.controller.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def train_env(self, x, y, epoch=1, verbose=1):
        self.env_model.train_model(x, y, batch_size=BATCH_SIZE, epoch=epoch, verbose=verbose)

    def train_r(self, x, y, epoch=1, verbose=0):
        self.r_model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.controller_target.predict(s)
        else:
            return self.controller.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, LATENT_DIM), target).flatten()

    def encode(self, s):
        encoded = np.asarray(self.encoder.predict(s))
        return encoded[0, 0, :]

    def updateTargetModel(self):
        self.controller_target.set_weights(self.controller.get_weights())
        

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ , d) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_, done) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        #if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            #self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def train_env(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(LATENT_DIM)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        x_env = np.zeros((len(batch), LATENT_DIM + actionCnt))
        y_env_s = np.zeros((len(batch), LATENT_DIM))
        y_env_r = np.zeros((len(batch), 1))
        y_env_d = np.zeros((len(batch), 1))

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3];
            done = o[4]

            x_env[i] = np.append(states[i], int2onehot(a, actionCnt))
            y_env_s[i] = states_[i]
            y_env_r[i] = r
            y_env_d[i] = done

        self.brain.train_env(x_env, y_env_s)
        self.brain.train_r(x_env, [y_env_r, y_env_d])

    def _getTargets(self, batch):
        batchLen = len(batch)

        no_state = np.zeros(LATENT_DIM)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=USE_TARGET)

        x = np.zeros((len(batch), LATENT_DIM))
        y = np.zeros((len(batch), self.actionCnt))
        errors = np.zeros(len(batch))

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            oldVal = t[a]
            # print (t)
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return x, y, errors

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
        self.brain.train_controller(x, y)
        for i in range(4):
            self.train_env()



#-------------------- ENVIRONMENT ---------------------
ss = None
selected = False
r_history = np.zeros(MAX_EPISODES)
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0
        global selected
        global ss
        global r_history
        if not selected:
            ss = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            selected = True
        #print (agent.brain.predictOne(ss))
        while True:
            if inspect: self.env.printState()

            sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            a = agent.act(sbar)
            s_, r, done = self.env.step(a)

            sbar_ = agent.brain.encode(np.reshape(s_, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            if done: # terminal state
                sbar_ = None

            agent.observe( (sbar, a, r, sbar_,done) )
            agent.replay()

            s = s_
            R += r
            r_history[episodes] = R

            if done:
                break

        print("Total reward:", R, ", episode: ", episodes)

#-------------------- MAIN ----------------------------
num_items = 3;
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)


try:
    while episodes < MAX_EPISODES:
        env.run(agent)
        episodes = episodes + 1
finally:
    ss=0
    agent.brain.controller.save("models/controller_601.h5")
    agent.brain.env_model.model.save("models/env_model_601.h5")
    agent.brain.r_model.save("models/r_model_601.h5")
    plt.plot(r_history)
    plt.show()
#env.run(agent, False)
