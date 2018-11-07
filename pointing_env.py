'''
This script load images and simulates an HRI envrionment by implementring a state machine. Each state in the FSM
is associated with a set of images, and the environmrnt ouputs one of the images at random for each state.
'''

import numpy as np
import random
from load_process_images import getImages
import matplotlib.pyplot as plt

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_STACK = 2

class PointingEnv:
    def __init__(self, num_items=3, use_all=True, val=False):
        self.state = None
        self.num_items = num_items
        self.purple, self.blue, self.orange, self.pu_bl, self.pu_or, self.bl_pu, self.bl_or, self.or_pu, self.or_bl,\
            self.pu_hand, self.bl_hand, self.or_hand = getImages(return_single=False ,use_all=use_all, val=val)


    def reset(self):
        #self.state is the internal state.
        #self.state = random.randint(0,1) #0 = b, 1 = g, 2 = b_only, 3 = g_only,
        self.state = random.choice([0, 1, 2])
        #self.state = 0
        return self._generateState()


    def step(self, action):
        assert action<self.getActSpaceSize() and action>=0, "action cannot exceed number of items +1 or be less than 0, action = %r" % action

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
        #x = stateArr.shape
        #print "state dims: "
        #print(x)
        #for i in self.item_list: print i
        #print self.item_list
        #print state
        #for i in state: print i

    def getStateSpaceSize(self):
        return ( IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def getActSpaceSize(self):
        return self.num_items+1


# testEnv = PointingEnv()
# img = testEnv.reset()
# plt.imshow(img)
# plt.show()
# ip = 0
# while(ip<4):
#     ip = int(raw_input('Enter action:'))
#     img, r, d = testEnv.step(ip)
#     if d:
#         img = testEnv.reset()
#     plt.imshow(img)
#     plt.show()

