import random, math, gym
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, Lambda, Concatenate
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from keras import backend as K
from keras import metrics
from pointing_env import PointingEnv
import matplotlib.pyplot as plt

# from tensorflow import contrib.distributions.MultivariateNormalDiag

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 4

ENV_LEARN_START = 0  # number of episodes before training env model starts`
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1  # 0.8
MIN_EPSILON = 1  # 0.0001
LAMBDA = 0.001  # speed of decay+
MAX_EPISODES = 1500
USE_TARGET = False
UPDATE_TARGET_FREQUENCY = 5
NUM_COMPONENTS = 3

epsilon_std = 1.0
BETA = 0.0
episodes = 0


def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot


def onehot2int(onehot):
    return np.argmax(onehot)


def mse(x, y):
    return ((x - y) ** 2).mean()


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.controller, self.env_model, self.encoder, self.controller_target, self.env_model_train, self.r_model \
            = self._createModel()

    def _createModel(self):
        encoder = load_model('models/encoder_105.h5')

        controller_input = Input(shape=(LATENT_DIM,), name='controller_input')
        controller_out = Dense(units=512, activation='relu')(controller_input)
        controller_out = Dense(units=256, activation='relu')(controller_out)
        # controller_out = Dense(units=32, activation='relu')(controller_out)
        # controller_out = Dense(units=16, activation='relu')(controller_out)
        controller_out = Dense(units=actionCnt, activation='linear')(controller_out)
        controller = Model(inputs=controller_input, outputs=controller_out)
        controller_opt = adam(lr=0.00025)
        controller.compile(loss='mse', optimizer='adam')
        controller.summary()

        # just copy the architecure
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        # def sampling(args):
        #     z_means, z_log_vars, coefficients = args
        #     #component = np.random.choice(np.linspace(0, NUM_COMPONENTS-1, NUM_COMPONENTS, dtype=int), p=coeff_arr[0, :])
        #     component = get_component_idx(K.random_uniform(shape=(1,), minval=0, maxval=1), coefficients)
        #     #z_mean = z_means[:, component*LATENT_DIM:(component+1)*LATENT_DIM]
        #     #z_log_var = z_log_vars[:, component*LATENT_DIM:(component+1)*LATENT_DIM]
        #     epsilon = K.random_normal(shape=(K.shape(z_means)[0], LATENT_DIM), mean=0.,
        #                               stddev=epsilon_std)
        #     return z_means + K.exp(z_log_vars / 2) * epsilon
        #
        # def get_component_idx(x, pdf):
        #     print(K.shape(pdf))
        #     selection = K.zeros(shape = (3,))
        #     accumulate = 0
        #     for i in range(0, NUM_COMPONENTS):
        #         accumulate += pdf[i]
        #         selection[i] = K.switch(K.greater_equal(accumulate, x), K.constant(1, shape=(1,)), K.constant(0,shape=(1,)))
        #     return K.argmax(selection, axis=1)

        def env_loss(y, out_params):
            # args in (y_true, y_pred) format
            # print(K.shape(out_params))
            means = out_params[:, 0:LATENT_DIM * NUM_COMPONENTS]
            log_vars = out_params[:, LATENT_DIM * NUM_COMPONENTS:LATENT_DIM * NUM_COMPONENTS * 2]
            coeffs = out_params[:, LATENT_DIM * NUM_COMPONENTS * 2:]
            loss = 0
            kl_loss = 0
            for i in range(0, NUM_COMPONENTS):
                mean = means[:, i * LATENT_DIM: (i + 1) * LATENT_DIM]
                log_var = log_vars[:, i * LATENT_DIM:(i + 1) * LATENT_DIM]
                coeff = coeffs[:, i]
                epsilon = 1e-8
                # epsilon = K.random_normal(shape=(K.shape(y)[0], LATENT_DIM), mean=0., stddev=0.05)
                var = K.exp(log_var)  # +epsilon
                # var = K.clip(var, min_value=epsilon, max_value=100)
                # var += epsilon
                std_dev = K.sqrt(var)
                # kl_loss += - 0.5 * K.sum(1 + log_var / K.log(2.) - K.square(mean) / (4.) - K.exp(log_var) / (4.),axis=-1)
                # std_prior = mean*0+math.sqrt(1)
                # var_prior = std_prior*std_prior
                # kl_loss += K.sum(K.log(std_prior/std_dev) + (var+K.square(mean))/2*var_prior -0.5, axis=-1)
                kl_loss += - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)

                def get_component_pdf(y, mean, var):
                    inv_covar = tf.matrix_diag(1 / var)
                    # inv_covar = K.clip(inv_covar, min_value=epsilon, max_value=100)
                    p = K.expand_dims(y - mean, axis=2)
                    p = K.permute_dimensions(p, (0, 2, 1))
                    p = K.batch_dot(p, inv_covar)
                    p = K.batch_dot(p, y - mean)
                    p = K.exp(-p / 2)
                    covar = tf.matrix_diag(var)
                    det_covar = K.cumprod(var, axis=1)
                    det_covar = K.sqrt(det_covar[:, LATENT_DIM - 1])
                    # det_covar = K.clip(det_covar, min_value=epsilon, max_value=100)
                    norm_const = 1  # math.sqrt(math.pow(2*math.pi, LATENT_DIM))
                    p = p / ((det_covar) * norm_const + epsilon)
                    # p = K.clip(p, min_value=0, max_value=1)
                    return p

                # p = get_component_pdf(y, mean, var)
                dist = tf.contrib.distributions.MultivariateNormalDiag(mean, std_dev, allow_nan_stats=False)
                p = dist.prob(y)
                # p = dist.prob(y + epsilon)
                # p = K.clip(p, min_value=0, max_value=1)
                # assert_op = tf.Assert(tf.less_equal(tf.reduce_max(p), 1.), [p])
                # with tf.control_dependencies([assert_op]):
                loss += coeff * p
            loss = -K.log(loss) + BETA * kl_loss / LATENT_DIM
            # loss = kl_loss
            # loss = -K.log(loss)
            # loss = K.sum(loss)
            # loss = K.sum(loss, axis=1)
            # x = K.batch_flatten(x)
            # x_decoded_mean = K.batch_flatten(x_decoded_mean)
            # xent_loss = metrics.mean_squared_error(x, x_decoded_mean)
            # xent_loss = LATENT_DIM*metrics.binary_crossentropy(x, x_decoded_mean)
            # kl_loss = - 0.5 * K.sum(1 + env_out_log_var - K.square(env_out_mean) - K.exp(env_out_log_var), axis=-1)
            # kl_loss = - 0.5 * K.sum(1 + env_out_log_var/K.log(2.) - K.square(env_out_mean)/(4.) - K.exp(env_out_log_var)/(4.), axis=-1)
            # return xent_loss# + BETA * kl_loss
            return K.mean(loss)

        env_model_input = Input(shape=(LATENT_DIM + actionCnt,), name='env_in')
        env_out = Dense(units=512, activation='relu', name='env_dense1')(env_model_input)
        env_out = Dense(units=256, activation='relu', name='env_dense2')(env_out)
        env_out = Dense(units=128, activation='relu', name='env_dense3')(env_out)
        # env_out = Dense(units=128, activation='relu', name='env_dense3')(env_out)
        env_out_mean = Dense(units=LATENT_DIM * NUM_COMPONENTS, name='env_out_mean')(env_out)
        env_out_log_var = Dense(units=LATENT_DIM * NUM_COMPONENTS, name='env_out_logvar')(env_out)
        env_out_coefficients = Dense(units=NUM_COMPONENTS, activation='softmax', name='env_out_coefficients')(env_out)
        # env_out = Lambda(sampling, output_shape=(LATENT_DIM,))([env_out_mean, env_out_log_var, env_out_coefficients])
        # env_model_train = Model(inputs=env_model_input, outputs=env_out)
        env_out_concat = Concatenate()([env_out_mean, env_out_log_var, env_out_coefficients])
        env_model = Model(inputs=env_model_input, outputs=[env_out_mean, env_out_log_var, env_out_coefficients])
        env_model_train = Model(inputs=env_model_input, outputs=env_out_concat)
        opt_env = adam(lr=0.00001)
        env_model_train.compile(loss=env_loss, optimizer=opt_env)
        env_model_train.summary()

        r_model_input = Input(shape=(LATENT_DIM + actionCnt,), name='r_in')
        r_model_out = Dense(units=512, activation='relu', name='r_dense1')(r_model_input)
        r_model_out = Dense(units=256, activation='relu', name='r_dense2')(r_model_out)
        r_out = Dense(units=1, name='r_out', activation='linear')(r_model_out)
        d_out = Dense(units=1, activation='sigmoid', name='d_out')(r_model_out)
        r_model = Model(r_model_input, [r_out, d_out])
        r_model.compile(loss='mse', optimizer='adam')

        return controller, env_model, encoder, controller_target, env_model_train, r_model

    def train_controller(self, x, y, epoch=1, verbose=0):

        self.controller.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def train_env(self, x, y, epoch=4, verbose=1):
        self.env_model_train.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def train_r(self, x, y, epoch=4, verbose=0):
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
        # return encoded[0, :]

    def updateTargetModel(self):
        self.controller_target.set_weights(self.controller.get_weights())


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ , d)
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


# -------------------- AGENT ---------------------------
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
            return random.randint(0, self.actionCnt - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_, done) format
        self.memory.add(sample)

        if USE_TARGET and (self.steps % UPDATE_TARGET_FREQUENCY == 0):
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        # self.epsilon = 0.1

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(LATENT_DIM)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=USE_TARGET)

        x = np.zeros((len(batch), LATENT_DIM))
        y = np.zeros((len(batch), self.actionCnt))

        x_env = np.zeros((len(batch), LATENT_DIM + actionCnt))
        # print 'xenv' , x_env.shape
        # y_env = np.zeros((len(batch), LATENT_DIM+2))
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
            # print 'sbar[i]', s_bar[i].shape
            # if s_ != None:
            # print(s_.shape)
            # print('s_', s_, states_[i])
            x_env[i] = np.append(states[i], int2onehot(a, actionCnt))
            # y_env[i] = np.append(states_[i], [r, done])
            y_env_s[i] = states_[i]  # - states[i]
            y_env_r[i] = r
            y_env_d[i] = done
            # print(x_env[i], y_env[i])

        self.brain.train_controller(x, y)

        # print(x_env, y_env)

        if episodes > ENV_LEARN_START:
            self.brain.train_env(x_env, y_env_s)
            self.brain.train_r(x_env, [y_env_r, y_env_d])
            # res = self.brain.env_model.predict(x_env)
            # means = res[0].flatten()
            # logvars = res[1].flatten()
            # coeffs = res[2].flatten()
            # print('coeffs = ', coeffs, 'means = ', means, 'vars = ', np.exp(logvars))


# -------------------- ENVIRONMENT ---------------------
ss = None
selected = False
r_history = np.zeros(MAX_EPISODES)


class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect=False):
        s = self.env.reset()
        R = 0
        global selected
        global ss
        global r_history
        if not selected:
            ss = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            selected = True
        # print (agent.brain.predictOne(ss))
        while True:
            if inspect: self.env.printState()

            sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            # print(sbar)

            a = agent.act(sbar)
            # print(a)
            s_, r, done = self.env.step(a)

            sbar_ = agent.brain.encode(np.reshape(s_, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            if done:  # terminal state
                sbar_ = None

            agent.observe((sbar, a, r, sbar_, done))
            agent.replay()

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
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

try:
    while episodes < MAX_EPISODES:
        env.run(agent)
        episodes = episodes + 1
finally:
    ss = 0
    agent.brain.controller.save("models/controller_310.h5")
    agent.brain.env_model.save("models/env_model_310.h5")
    agent.brain.r_model.save("models/r_model_310.h5")
    plt.plot(r_history)
    plt.show()
# env.run(agent, False)
