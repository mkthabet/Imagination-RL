import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout
from keras.optimizers import *
from keras.models import Model
from imgEnv import *

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

ENV_LEARN_START = 100   #number of episodes before training env model starts

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.model, self.env_model, self.dqn_head_model, self.conv_model, self.dqn_target = self._createModel()

    def _createModel(self):
        img_input = Input(shape = self.stateCnt)
        conv = Conv2D(32, (8, 8), strides=(4,4), activation='relu', data_format='channels_last')(img_input)
        conv1 = Conv2D(64, (4, 4), strides=(2,2), activation='relu')(conv)
        conv2layer = Conv2D(64, (3, 3), activation='relu')
        conv_out = conv2layer(conv1)
        conv_out_layer = Flatten()
        conv_out = conv_out_layer(conv_out)
        conv_out_dense_layer = Dense(units=512, activation='linear', name = 'conv_out_dense')
        conv_out = conv_out_dense_layer(conv_out)
        conv_model = Model(img_input, conv_out)
        opt = RMSprop(lr=0.00025)
        conv_model.compile(loss='mse', optimizer=opt)

        #dqn_head_input = Input(shape=conv_out_layer.output_shape)
        dqn_head_input = Input(shape = conv_out_dense_layer.output_shape)
        dqn_out = Dense(units=512, activation='relu')(dqn_head_input)
        dqn_out = Dense(units=actionCnt, activation='linear')(dqn_out)
        dqn_head_model = Model(inputs=dqn_head_input, outputs=dqn_out)
        dqn_head_model.compile(loss='mse', optimizer=opt)

        q_out = dqn_head_model(conv_out)
        dqn_model = Model(img_input,q_out)
        dqn_model.compile(loss='mse', optimizer=opt)

        dqn_target = Model(img_input,q_out)
        dqn_target.compile(loss='mse', optimizer=opt)

        #print conv_out_layer.output_shape

        #env_in_shape = (conv_out_layer.output_shape[0], conv_out_layer.output_shape[1]+1)
        env_in_shape = (conv_out_dense_layer.output_shape[0], conv_out_dense_layer.output_shape[1] +1)
        env_model_input = Input(shape=env_in_shape, name = 'env_in')
        #print 'env in shape', env_in_shape
        env_out = Dense(units=512, activation='relu', name = 'env_dense1')(env_model_input)
        #env_dropout1 = Dropout(0.5)
        #env_out = env_dropout1(env_out)
        env_out = Dense(units=256, activation='relu', name = 'env_dense2')(env_out)
        #env_dropout2 = Dropout(0.5)
        #env_out = env_dropout2(env_out)
        env_out = Dense(units=conv_out_dense_layer.output_shape[1]+2, activation='linear', name = 'env_out')(env_out)
        env_model = Model(inputs=env_model_input, outputs=env_out)
        opt_env = RMSprop(lr=0.00025)
        env_model.compile(loss='mse', optimizer=opt_env)

        return dqn_model, env_model, dqn_head_model, conv_model, dqn_target

    def train(self, x, y, epoch=1, verbose=0):

        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def train_env(self, x, y, epoch=1, verbose=0):
        self.env_model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.dqn_target.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3), target).flatten()

    def get_s_bar(self, s):
        return self.conv_model.predict(s)

    def updateTargetModel(self):
        self.dqn_target.set_weights(self.model.get_weights())
        

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
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 0.6
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 50

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

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        a_vec = numpy.array([ o[1] for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=True)
        s_bar = agent.brain.get_s_bar(states)
        #print 'sbar' , s_bar.shape
        s_bar_= agent.brain.get_s_bar(states_)

        x = numpy.zeros((len(batch), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        y = numpy.zeros((len(batch), self.actionCnt))

        x_env = numpy.zeros((len(batch), s_bar.shape[1]+1))
        #print 'xenv' , x_env.shape
        y_env = numpy.zeros((len(batch), s_bar.shape[1]+2))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]; done = o[4]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t
            #print 'sbar[i]', s_bar[i].shape
            x_env[i] = np.append(s_bar[i], a)
            y_env[i] = np.append(s_bar_[i], [r, done])

        self.brain.train(x, y)

        if episodes>ENV_LEARN_START:
            #print 'expand dims', np.expand_dims(x_env, axis = 0).shape
            self.brain.train_env(np.expand_dims(x_env,axis = 0),np.expand_dims(y_env,axis = 0))
            #self.brain.train_env(x_env, y_env)


#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0 
        global sortedCnt
        while True:         
            if inspect: self.env.printState()   
            a = agent.act(s)

            s_, r, done = self.env.step(a)
            

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_,done) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R, ", episode: ", episodes)

#-------------------- MAIN ----------------------------
num_items = 2;
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

episodes = 0
MAX_EPISODES = 1000

try:
    while episodes < MAX_EPISODES:
        env.run(agent)
        episodes = episodes + 1
finally:
    agent.brain.model.save("models/model_11.h5")
    agent.brain.env_model.save("models/env_model_11.h5")
    agent.brain.dqn_head_model.save("models/dqn_head_model_11.h5")
    agent.brain.conv_model.save("models/conv_model_11.h5")
#env.run(agent, False)
