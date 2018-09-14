

import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from imgEnv_val import *

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

sortedCnt = 0

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model, self.conv_model, self.dqn_head_model = self._createModel()
        #self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = load_model("models/model_1.h5")
        conv_model = load_model("models/conv_model_1.h5")
        dqn_head_model = load_model("models/dqn_head_model_1.h5")

        return model, conv_model, dqn_head_model

    def train(self, x, y, epoch=1, verbose=0):

        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        # print "shape"
        # print(s.shape)
        #print "PREDICT:"
        #print self.model.predict(s)
        return self.model.predict(s)

    def predictOne(self, s):
        #print("state:", s)
      #  print " predictone:"
        #print self.predict(s.reshape(1, self.stateCnt)).flatten()
        return self.predict(s.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)).flatten()
        

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
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

MAX_EPSILON = 0.5
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

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

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        x = numpy.zeros((len(batch), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        y = numpy.zeros((len(batch), self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0
        while True:
            #cv2.imshow('test', s)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            a = agent.act(s)
            print a
            s_, r, done = self.env.step(a)

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
num_items = 2;
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

episodes = 0
MAX_EPISODES = 1000

env.run(agent)
    #agent.brain.model.save("point_3.h5")
#env.run(agent, False)
