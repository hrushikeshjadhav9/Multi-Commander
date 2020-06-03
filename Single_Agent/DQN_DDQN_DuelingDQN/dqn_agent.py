"""
DQN and Double DQN agent implementation using Keras
"""

import random # To generate random numbers
import numpy as np # To do math in python
from collections import deque # To build a queue for storing experience of agent
from keras.models import Sequential # To build a DNN
from keras.layers import Dense # Layers in the DNN
from keras.optimizers import Adam # Gradient descent algorithm
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import os # To interact with the OS

from tensorflow.keras.callbacks import TensorBoard # New
import time

# os.environ is a python dictionary containing keys as environment variables
# and values as env variable values.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Using 2 GPUs. GPU 1 and GPU 2.

"""
Sessions are gone in TensorFlow 2.
There is one global runtime in the background that executes all computation,
whether run eagerly or as a compiled tf.function.
"""
# tf.compat.v1.keras.backend.set_session(tf.Session(config=tf.ConfigProto(\
# device_count={'gpu':0})))


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        # self.writer = tf.summary.FileWriter(self.log_dir)
        # self.writer = tf.contrib.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir) # New

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        # self._write_logs(stats, self.step)
        # tf.summary.scalar('loss',stats['loss'], step=self.step) # New
        tf.summary.scalar('reward_avg',stats['reward_avg'], step=self.step) # New



class DQNAgent: # Blueprint of DQN Agent.
    def __init__(self, config): # Initializing attributes of the agent.
        self.state_size = config['state_size'] # Dimension of state vector.
        self.action_size = config['action_size'] # Dimension of action vector.
        self.memory = deque(maxlen=3000) # Queue to store (s,a,_s,r) experience\ tuples.
        self.gamma = 0.95 # Discounting rate.
        self.epsilon = 1 # Probability that agent exploits current knowledge.
        self.epsilon_min = 0.001 # epsilon shouldn't fall below 0.1 after decaying.
        self.epsilon_decay = 0.99975 # rate at which epsilon falls.
        self.learning_rate = 0.001 # rate at which agent / model learns good behavior.
        self.update_target_freq = 5 # target DNN updates every 5 episodes.
        self.batch_size = config['batch_size'] # Number of experience tuples used at once to train the agent.
        self.model = self._build_model() # Builds online DNN.
        self.target_model = self._build_model() # Builds target DNN.
        self.update_target_network() # Updates target DNN with learnings / weights of online DNN.
        intersection_id = list(config['lane_phase_info'].keys())[0]
        self.phase_list = config['lane_phase_info'][intersection_id]['phase']

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(int(time.time())))

    def _build_model(self):
        """
        It defines and builds our DNN. Returns the model.

        We use linear activation function because Q values can be real numbers.
        Activation functions like sigmoid and tanh restrict the values to be in
        a certain range and impede flexibility which is provided by linear activation
        function which helps in our use case.
        """
        model = Sequential()
        model.add(Dense(40, input_dim = self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        """ Fetch learnt weights from online model. Duplicate them on the target model """
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        """ enqueue experience sequence in history of sequences """
        action = self.phase_list.index(action)
        self.memory.append((state, action, reward, next_state)) # Appending a sequence tuple to a queue of tuples.

    def choose_action(self, state):
        """ Agent decides which action to take in the current state """
        if np.random.rand() >= self.epsilon: # random number from [0,1] >= 0.9.
            return random.randrange(self.action_size) # returns value from [0,9].
        act_values = self.model.predict(state) # Finds Q values for all actions in a state.
        return np.argmax(act_values[0]) # returns action corresponding to maximum Q value.

    def replay(self, terminal_state): # Trains target DNN from experiences of online DNN.
        minibatch = random.sample(self.memory, self.batch_size) # Form a minibatch to learn from.
        X = []
        Y = []
        for state, action, reward, next_state in minibatch:
            # Why target_model is used for prediction here and online model is used for prediction underneath
            target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])) # new_q
            target_f = self.model.predict(state) # current_qs_list
            target_f[0][action] = target
            X.append(state)
            Y.append(target_f)

        # Fit on all samples as one batch, log only on terminal state.
        # Call tensorboard on reaching terminal state.
        self.model.fit(np.array(X)/255, np.array(Y), batch_size=self.batch_size, epochs=10, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # self.model.fit(state, target_f, epochs=10, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        print("model saved:{}".format(name))


class DDQNAgent(DQNAgent):
    def __init__(self, config):
        super(DDQNAgent, self).__init__(config)

    def replay(self, terminal_state):
        minibatch = random.sample(self.memory, self.batch_size)
        X = []
        Y = []
        for state, action, reward, next_state in minibatch:
            target_f = self.model.predict(state)
            actions_for_next_state = np.argmax(self.model.predict(next_state)[0])
            target = (reward + self.gamma * self.target_model.predict(mext_state)[0][actions_for_next_state])
            target_f[0][action] = target
            X.append(state)
            Y.append(target_f)
            # self.model.fit(state, target_f, epochs=1, verbose=0)
        self.model.fit(np.array(X)/255, np.array(Y), batch_size=self.batch_size, epochs=10, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
