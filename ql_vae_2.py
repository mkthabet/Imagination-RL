import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, Lambda
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from keras import backend as K
from keras import metrics
from imgEnv import *
import matplotlib.pyplot as plt

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 3

ENV_LEARN_START = 200   #number of episodes before train_controllering env model starts`
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 0.8
MIN_EPSILON = 0.0001
LAMBDA = 0.001      # speed of decay
MAX_EPISODES = 1200
USE_TARGET = False
UPDATE_TARGET_FREQUENCY = 5

epsilon_std = 1.0
BETA = 8
episodes = 0

def mse(x,y):
    return ((x-y)**2).mean()

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.controller, self.env_model, self.encoder, self.controller_target, self.env_model_train, self.r_model = self._createModel()

    def _createModel(self):
        encoder = load_model('models/encoder_12.h5')

        controller_input = Input(shape=(LATENT_DIM,), name='controller_input')
        controller_out = Dense(units=512, activation='relu')(controller_input)
        controller_out = Dense(units=256, activation='relu')(controller_out)
        #controller_out = Dense(units=32, activation='relu')(controller_out)
        #controller_out = Dense(units=16, activation='relu')(controller_out)
        controller_out = Dense(units=actionCnt, activation='linear')(controller_out)
        controller = Model(inputs=controller_input, outputs=controller_out)
        opt = RMSprop(lr=0.00025)
        controller.compile(loss='mse', optimizer='adam')
        controller.summary()

        #just copy the architecure
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM), mean=0.,
                                      stddev=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        def env_loss(x, x_decoded_mean):
            x = K.batch_flatten(x)
            x_decoded_mean = K.batch_flatten(x_decoded_mean)
            xent_loss = (LATENT_DIM) * metrics.binary_crossentropy(x, x_decoded_mean)
            # xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
            #kl_loss = - 0.5 * K.sum(1 + env_out_log_var - K.square(env_out_mean) - K.exp(env_out_log_var), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + env_out_log_var/K.log(2.) - K.square(env_out_mean)/(4.) - K.exp(env_out_log_var)/(4.), axis=-1)
            return xent_loss + BETA * kl_loss

        env_model_input = Input(shape=(LATENT_DIM+1,), name = 'env_in')
        env_out = Dense(units=512, activation='relu', name = 'env_dense1')(env_model_input)
        env_out = Dense(units=256, activation='relu', name = 'env_dense2')(env_out)
        #env_out = Dense(units=128, activation='relu', name='env_dense3')(env_out)
        env_out_mean = Dense(units=LATENT_DIM, name = 'env_out_mean')(env_out)
        env_out_log_var = Dense(units=LATENT_DIM, name = 'env_out_logvar')(env_out)
        #r_out = Dense(units=1, name='r_out')(env_out)
        #d_out = Dense(units = 1, activation= 'sigmoid', name = 'd_out')(env_out)
        env_out = Lambda(sampling, output_shape=(LATENT_DIM,))([env_out_mean, env_out_log_var])
        env_model_train = Model(inputs=env_model_input, outputs=env_out)
        env_model = Model(inputs=env_model_input, outputs=[env_out_mean, env_out_log_var])
        opt_env = RMSprop(lr=0.00025)
        env_model_train.compile(loss=env_loss, optimizer='adam')

        r_model_input = Input(shape=(LATENT_DIM+1,), name = 'r_in')
        r_model_out = Dense(units=512, activation='relu', name = 'r_dense1')(r_model_input)
        r_model_out = Dense(units=256, activation='relu', name = 'r_dense2')(r_model_out)
        r_out = Dense(units=1, name='r_out')(r_model_out)
        d_out = Dense(units = 1, activation= 'sigmoid', name = 'd_out')(r_model_out)
        r_model = Model(r_model_input, [r_out, d_out])
        r_model.compile(loss='mse', optimizer='adam')

        return controller, env_model, encoder, controller_target, env_model_train, r_model

    def train_controller(self, x, y, epoch=1, verbose=0):

        self.controller.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

    def train_env(self, x, y, epoch=4, verbose=0):
        self.env_model_train.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

    def train_r(self, x, y, epoch=4, verbose=0):
        self.r_model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

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
        #return encoded[0, :]

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
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_, done) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        #self.epsilon = 0.1

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(LATENT_DIM)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=USE_TARGET)

        x = numpy.zeros((len(batch), LATENT_DIM))
        y = numpy.zeros((len(batch), self.actionCnt))

        x_env = numpy.zeros((len(batch), LATENT_DIM+1))
        #print 'xenv' , x_env.shape
        #y_env = numpy.zeros((len(batch), LATENT_DIM+2))
        y_env_s = numpy.zeros((len(batch), LATENT_DIM))
        y_env_r = np.zeros((len(batch), 1))
        y_env_d = np.zeros((len(batch), 1))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]; done = o[4]
            
            t = p[i]
           # print (t)
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t
            #print 'sbar[i]', s_bar[i].shape
            #if s_ != None:
                #print(s_.shape)
            #print('s_', s_, states_[i])
            x_env[i] = np.append(states[i], a)
            #y_env[i] = np.append(states_[i], [r, done])
            y_env_s[i] = states_[i] - states[i]
            y_env_r[i] = r
            y_env_d[i] = done
            #print(x_env[i], y_env[i])

        self.brain.train_controller(x, y)

        #print(x_env, y_env)

        if episodes>ENV_LEARN_START:
            self.brain.train_env(x_env, y_env_s)
            self.brain.train_r(x_env, [y_env_r, y_env_d])


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
            #print(sbar)

            a = agent.act(sbar)
           # print(a)
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
num_items = 2;
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
    agent.brain.controller.save("models/controller_212.h5")
    agent.brain.env_model.save("models/env_model_212.h5")
    agent.brain.r_model.save("models/r_model_212.h5")
    plt.plot(r_history)
    plt.show()
#env.run(agent, False)
