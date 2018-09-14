import numpy as np
import random
import cv2
import os

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

def processImage( img ):
    #rgb = None
    rgb = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return rgb
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance
    #o = gray.astype('float32') / 128 - 1    # normalize
    #return o

class PointingEnv:
    def __init__(self, num_items = 2):
        self.num_items = num_items
        self.act_space_size = self.num_items
        self.b_imgs, self.g_imgs, self.b_only, self.g_only, self.b_hand, self.g_hand = [], [], [], [], [], []
        for filename in os.listdir('validation/b'):
            img = cv2.imread(os.path.join('validation/b',filename))
            if img is not None:
                self.b_imgs.append(processImage(img))
        for filename in os.listdir('validation/g'):
            img = cv2.imread(os.path.join('validation/g',filename))
            if img is not None:
                self.g_imgs.append(processImage(img))
        for filename in os.listdir('validation/b_only'):
            img = cv2.imread(os.path.join('validation/b_only', filename))
            if img is not None:
                self.b_only.append(processImage(img))
        for filename in os.listdir('validation/g_only'):
            img = cv2.imread(os.path.join('validation/g_only', filename))
            if img is not None:
                self.g_only.append(processImage(img))
        for filename in os.listdir('validation/b_hand'):
            img = cv2.imread(os.path.join('validation/b_hand', filename))
            if img is not None:
                self.b_hand.append(processImage(img))
        for filename in os.listdir('validation/g_hand'):
            img = cv2.imread(os.path.join('validation/g_hand', filename))
            if img is not None:
                self.g_hand.append(processImage(img))


    def reset(self):
        #self.state is the internal state.
        self.state = random.randint(0,1) #0 = b, 1 = g, 2 = b_only, 3 = g_only,
        return self._generateState()


    def step(self, action):
        assert action<self.getActSpaceSize() and action>=0, "action cannot exceed number of items +1 or be less than 0, action = %r" % action

        done = 0

        if self.state == 0:
            if action == 0:
                #self.state = random.choice([3, 5])
                self.state = 3
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 1:
            if action == 1:
                #self.state = random.choice([2, 4])
                self.state = 4
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 2:
            if action == 1:
                self.state = 0
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 3:
            if action == 0:
                self.state = 1
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 4:
            if action == 2:
                done = 1
                reward = 5
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 5:
            if action == 2:
                done = 1
                reward = 5
            else:
                done = 1
                reward = -1
                print 'mistake.....'

        return self._generateState(), reward, done

    def _generateState(self):
        # 0 = b, 1 = g, 2 = b_only, 3 = g_only, 4 = b_hand, 5 = g_hand
        #print 'state = ' , self.state
        if self.state == 0:
            return random.choice(self.b_imgs)
        elif self.state == 1:
            return random.choice(self.g_imgs)
        elif self.state == 2:
            return random.choice(self.b_only)
        elif self.state == 3:
            return random.choice(self.g_only)
        elif self.state == 4:
            return random.choice(self.b_hand)
        elif self.state == 5:
            return random.choice(self.g_hand)


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
# cv2.imshow('test', img)
# cv2.waitKey(0)
# ip = 0
# while(ip<4):
#     ip = int(raw_input('Enter action:'))
#     img, r, d = testEnv.step(ip)
#     cv2.imshow('test', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

