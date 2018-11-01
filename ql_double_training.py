import random, math, gym
import numpy as np

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout
from keras.optimizers import *
from keras.models import Model, load_model, model_from_json
from pointing_env import PointingEnv
import matplotlib.pyplot as plt
from mdn import MDN
from model_tester import test_model

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 8

RE_MEMORY_CAPACITY = 10000
IM_MEMORY_CAPACITY = 200
ENV_BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 0.6  # 0.8
MIN_EPSILON = 0.0001  # 0.0001
LAMBDA = 0.1  # speed of decay+
MAX_EPISODES = 100
USE_TARGET = False
UPDATE_TARGET_FREQUENCY = 5
NUM_COMPONENTS = 48
R_ENV = 32 #number of env training batches for each episode
R_C = 4    #number of updates per step for controller
epsilon_std = 1.0
BETA = 0.0
episodes = 0
SIGMA_NOISE = 0.15
actionCnt = 0

ENV_LEARN_START = 40  # number of episodes before training env model starts`
I_D = 4     #imaginary rollout depth (length of rollout)
I_B = 10    #imaginary rollout breadth (number of rollouts)
I_START = 50    # episode at which imaginary training starts
MEM_BATCHSIZE = 128      #total batch size for replay
IM_PERCENT = 0.5        #percentage of total batch size that is imaginary transitions
IM_BATCHSIZE = int(round(MEM_BATCHSIZE*IM_PERCENT))
RE_BATCHSIZE = MEM_BATCHSIZE - IM_BATCHSIZE


def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot


class EnvironmentModel:
    def __init__(self):
        self.z = None
        self.env_model = MDN(num_components=NUM_COMPONENTS, in_dim=LATENT_DIM + actionCnt, out_dim=LATENT_DIM)
        self.r_model = self._createModel()
        self.decoder = load_model("models/decoder_208.h5")
        self.statecnt = LATENT_DIM

    def _createModel(selfself):
        r_model_input = Input(shape=(LATENT_DIM + actionCnt,), name='r_in')
        r_model_out = Dense(units=512, activation='relu', name='r_dense1')(r_model_input)
        r_model_out = Dense(units=256, activation='relu', name='r_dense2')(r_model_out)
        r_out = Dense(units=1, name='r_out', activation='linear')(r_model_out)
        d_out = Dense(units=1, activation='sigmoid', name='d_out')(r_model_out)
        r_model = Model(r_model_input, [r_out, d_out])
        r_model.compile(loss='mse', optimizer='adam')
        return r_model

    def init_model(self, z):
        self.z = z
        # print('new episode ')
        # plt.imshow(decoded.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        # plt.show()

    def step(self, a):
        s_a = np.append(self.z, int2onehot(a, actionCnt))
        mu, sigma, pi = self.env_model.get_dist_params(s_a.reshape(1, -1))
        component = np.random.choice(np.arange(0, NUM_COMPONENTS, 1), p=pi.flatten())
        mu = mu[0, component, :]
        self.z = mu
        # self.z = model_out
        r_out = self.r_model.predict(s_a.reshape((1, s_a.size)))
        r = r_out[0].flatten()
        done = r_out[1].flatten()

        return self.z, r, done

    def train_env(self, x, y, epoch=1, verbose=0):
        self.env_model.train_model(x, y, batch_size=ENV_BATCH_SIZE, epoch=epoch, verbose=verbose)

    def train_r(self, x, y, epoch=1, verbose=0):
        self.r_model.fit(x, y, batch_size=ENV_BATCH_SIZE, epochs=epoch, verbose=verbose)


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.env_model = EnvironmentModel()
        self.controller, self.encoder, self.controller_target = self._createModel()

    def _createModel(self):
        encoder = load_model('models/encoder_208.h5')

        controller_input = Input(shape=(LATENT_DIM,), name='controller_input')
        controller_out = Dense(units=512, activation='relu')(controller_input)
        controller_out = Dense(units=256, activation='relu')(controller_out)
        # controller_out = Dense(units=32, activation='relu')(controller_out)
        # controller_out = Dense(units=16, activation='relu')(controller_out)
        controller_out = Dense(units=actionCnt, activation='linear')(controller_out)
        controller = Model(inputs=controller_input, outputs=controller_out)
        controller_opt = adam(lr=0.00025)
        controller.compile(loss='mse', optimizer='adam')
        #controller.summary()

        # just copy the architecture
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        return controller, encoder, controller_target

    def train(self, x, y, epoch=1, verbose=0):
        self.controller.fit(x, y, batch_size=MEM_BATCHSIZE, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.controller_target.predict(s)
        else:
            return self.controller.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, LATENT_DIM), target).flatten()

    def encode(self, s):
        p_z = np.asarray(self.encoder.predict(s))
        return p_z[0, 0, :]

    def updateTargetModel(self):
        self.controller_target.set_weights(self.controller.get_weights())

    def train_env(self, x, y, epoch=1, verbose=0):
        self.env_model.train_env(x, y, epoch=epoch, verbose=verbose)

    def train_r(self, x, y, epoch=1, verbose=0):
        self.env_model.train_r(x, y, epoch=epoch, verbose=verbose)



# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ , d)

    def __init__(self, capacity):
        self.samples = []
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- AGENT ---------------------------
class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(RE_MEMORY_CAPACITY)
        self.imaginary_memory = Memory(IM_MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
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
        if not imaginary:
            self.steps += 1
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def train_env(self):
        batch = self.memory.sample(ENV_BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(LATENT_DIM)

        #states = np.array([o[0] for o in batch])
        #states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        states = np.array([(o[0] + np.random.normal(loc=0, scale=SIGMA_NOISE, size=LATENT_DIM)) for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3] + np.random.normal(loc=0, scale=SIGMA_NOISE, size=LATENT_DIM)) for o in batch])

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

            x_env[i] = np.append(states[i], int2onehot(a, actionCnt))
            y_env_s[i] = states_[i]
            y_env_r[i] = r
            y_env_d[i] = done

        self.brain.train_env(x_env, y_env_s)
        self.brain.train_r(x_env, [y_env_r, y_env_d])

    def replay(self, imaginary=False):
        batch = self.memory.sample(RE_BATCHSIZE)
        if imaginary:
            batch = batch + self.imaginary_memory.sample(IM_BATCHSIZE)

        batchLen = len(batch)
        no_state = np.zeros(LATENT_DIM)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=USE_TARGET)

        x = np.zeros((len(batch), LATENT_DIM))
        y = np.zeros((len(batch), self.actionCnt))

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

        self.brain.train(x, y)

# -------------------- ENVIRONMENT ---------------------

r_history = np.zeros(MAX_EPISODES)

class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent):
        s = self.env.reset()
        R = 0
        global r_history
        imaginary = False   #flag to start using imaginary rollouts for training
        if episodes > I_START:
            #print('Imaginary training started...')
            imaginary = True    #start using imaginary rollouts
        # TODO: decide if imaginary or not
        while True:
            z = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            a = agent.act(z)
            # print('a = ', a)
            if imaginary:
                for i_b in range(I_B):
                    agent.brain.env_model.init_model(z)
                    zhat = z
                    for i_d in range(I_D):
                        zhat_, rhat, donehat = agent.brain.env_model.step(a)
                        rhat, donehat = round(rhat), round(donehat)
                        if donehat == 1:
                            zhat_ = None
                        agent.observe((zhat, a, rhat, zhat_, donehat), imaginary=True)
                        if donehat == 1:
                            break
                        zhat = zhat_
            s_, r, done = self.env.step(a)
            z_ = agent.brain.encode(np.reshape(s_, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))

            if done:  # terminal state
                z_ = None

            agent.observe((z, a, r, z_, done), imaginary=False)
            if (episodes > I_START) and (episodes >= ENV_LEARN_START):
                for i in range(R_ENV):
                    agent.train_env()

            for i in range(R_C):
                agent.replay(imaginary=imaginary)

            s = s_
            R += r
            r_history[episodes] = R

            if done:
                break

        print("Total reward:", R, ", episode: ", episodes)


# -------------------- MAIN ----------------------------
num_items = 3;
env = Environment(num_items)

stateCnt = env.env.getStateSpaceSize()
global actionCnt
actionCnt = env.env.getActSpaceSize()


max_runs = 10
runs = 0
done_counts = []
while runs < max_runs:
    episodes = 0
    agent = Agent(stateCnt, actionCnt)
    print("training run ", runs+1)
    try:
        while episodes < MAX_EPISODES:
            env.run(agent)
            episodes = episodes + 1
    finally:
        ss = 0  # blah blah
        agent.brain.env_model.env_model.model.save("models/env_model_2001.h5")
        agent.brain.env_model.r_model.save("models/r_model_2001.h5")
        agent.brain.controller.save('models/controller_2001.h5')
        print("testing run ", runs+1)
        done_counts.append(test_model())
        runs += 1
        #plt.plot(r_history)
        #plt.show()
done_counts = np.asarray(done_counts)
print("average = ", done_counts.mean(), "max = ", done_counts.max(), "min = ", done_counts.min(), "sigma = ", np.std(done_counts))
# env.run(agent, False)
