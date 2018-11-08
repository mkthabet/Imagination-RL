'''
This script load images and simulates an HRI envrionment by implementring a state machine. Each state in the FSM
is associated with a set of images, and the environmrnt ouputs one of the images at random for each state.
'''

import numpy as np
import random
from load_process_images import getImages
from enum import Enum
import matplotlib.pyplot as plt

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_STACK = 2

class ArrowEnv:
    def __init__(self, num_items=3, use_all=True, val=False, stochastic_gestures=False, stochastic_dynamics=False):
        self.arrow_state = []
        self.gest_state = None
        self.num_items = num_items
        self.stoch_gest = stochastic_gestures
        self.stoch_dyn = stochastic_dynamics
        self.stoch_gest_prob = 0.1
        #TODO: write getImages()
        #self.purple, self.blue, self.orange, self.pu_bl, self.pu_or, self.bl_pu, self.bl_or, self.or_pu, self.or_bl,\
            #self.pu_hand, self.bl_hand, self.or_hand = getImages(return_single=False ,use_all=use_all, val=val)


    def reset(self):
        #self.arrow_state is the internal state.
        # numbers in the arrow_state list are the states of the arrows
        # arrow states key: 0 = U, 1 = L, 2 = D, 3 = R
        # gest_state is which arrow is being pointed to
        self.arrow_state = []
        self.arrow_state = random.sample(range(self.num_items), 3)
        self.gest_state = random.choice(range(self.num_items))
        while self._isSolved(): # avoid having the initial state already solved on reset
            self.gest_state = random.choice(range(self.num_items))
        return self._generateState()


    def step(self, action):
        #actions key: each 2 successive values represent rotating an object counerCW or CW respectively.
        #exanple: 0 = rotate item 0 CCW, 1 = rotate item 1 CW, 2 = rotate item 1 CCW etc...

        assert action<self.getActSpaceSize() and action>=0, \
            "action cannot exceed num_items*2 or be less than 0, action = %r" % action

        done = 0

        if action == 0:
            self.arrow_state[0] = (self.arrow_state[0] + 1) % (self.num_items + 1)
        elif action == 1:
            self.arrow_state[0] = (self.arrow_state[0] - 1) % (self.num_items + 1)
        elif action == 2:
            self.arrow_state[1] = (self.arrow_state[1] + 1) % (self.num_items + 1)
        elif action == 3:
            self.arrow_state[1] = (self.arrow_state[1] - 1) % (self.num_items + 1)
        elif action == 4:
            self.arrow_state[2] = (self.arrow_state[2] + 1) % (self.num_items + 1)
        elif action == 5:
            self.arrow_state[2] = (self.arrow_state[2] - 1) % (self.num_items + 1)

        # now compute reward
        if len(self.arrow_state) > len(set(self.arrow_state)):  # non-unique config
            reward = -10
            done = 1
            print("non-unique config!")
        elif self._isSolved():   # only arrow pointed to is up
            reward = 50
            done = 1
            print("Solved!")
        else:
            reward = -1

        # pointing has a random chance to change
        if self.stoch_gest and (random.random() <= self.stoch_gest_prob) and (done != 1):
            gest_aslist = []
            gest_aslist.append(self.gest_state)
            self.gest_state = random.choice(list(set(range(self.num_items)) - set(gest_aslist)))
            # set difference makes sure the new gesture is different

        return self._generateState(), reward, done

    def _generateState(self):
        # returns observable state
        state = list(self.arrow_state)
        return state.append(self.gest_state)

    def printState(self):
        print ("State: " + self._getStateString())

    def getStateSpaceSize(self):
        return ( IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def getActSpaceSize(self):
        return self.num_items*2

    def _isSolved(self):
        return self.arrow_state[self.gest_state] == 0

    def _getStateString(self):
        state_str = ''
        for i in range(len(self.arrow_state)):
            if self.arrow_state[i] == 0:
                state_str += 'U'
            elif self.arrow_state[i] == 1:
                state_str += 'L'
            elif self.arrow_state[i] == 2:
                state_str += 'D'
            elif self.arrow_state[i] == 3:
                state_str += 'R'
        state_str += str(self.gest_state)
        return state_str

testEnv = ArrowEnv(stochastic_gestures=True)
s = testEnv.reset()
testEnv.printState()
ip = 0
while(ip <= testEnv.getActSpaceSize()):
    ip = int(raw_input('Enter action:'))
    s, r, d = testEnv.step(ip)
    testEnv.printState()
    if d:
        s = testEnv.reset()
        print ("resetting...")
        testEnv.printState()

