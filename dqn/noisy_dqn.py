import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
import math
from collections import deque

tf.get_logger().setLevel('ERROR')
episodes = 1000
episode_rewards = []
step_limit = 200
memory_size = 100000
env = gym.make('CartPole-v1')
env.seed(777)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

saveFileName = 'noisynet'


# os.environ['CUDA_VISIBLE_DEVICES']='0'
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpus) > 0
# tf.config.experimental.set_memory_growth(gpus[0], True)

# leaves contain priorities for every experience.A data array containing the experiences points to the leaves.
# priorities are determined due to their TD error.
# Updating the tree and sampling will be really efficient (O(log n)).
# the value of root node is the sum of its child nodes
class SumTree:
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, priority, data):
        # the overall nodes is capacity(leaves) + capacity - 1
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        changed_value = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # if index = 6, then index = 2 and index = 0 in tree will add changed_value
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += changed_value

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    # an alternative method for getter and setter
    # we could directly use self.total_priority instead of self.get_total_priority()
    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


# class NoisyLinear(tf.keras.layers.Layer):
#     def __init__(self, out_features: int, in_features, std_init: float = 0.5):
#         """Initialization."""
#         super(NoisyLinear, self).__init__()
#         self.out_features = out_features
#         self.std_init = std_init
#
#         self.in_features = in_features
#         self.weight_mu = self.add_weight(name='weight_mu', shape=[self.in_features, self.out_features], trainable=True)
#         self.weight_sigma = self.add_weight(name='weight_sigma', shape=[self.in_features, self.out_features], trainable=True)
#         self.weight_epsilon = self.add_weight(name='weight_epsilon', shape=[self.in_features, self.out_features])
#
#         self.bias_mu = self.add_weight(name='bias_mu', shape=[self.out_features,], trainable=True)
#         self.bias_sigma = self.add_weight(name='bias_sigma', shape=[self.out_features,], trainable=True)
#         self.bias_epsilon = self.add_weight(name='bias_epsilon', shape=[self.out_features,])
#
#         self.reset_parameters()
#         self.reset_noise()
#
#     def reset_parameters(self):
#         """Reset trainable network parameters (factorized gaussian noise)."""
#
#         self.weight_mu = tf.keras.backend.random_uniform(self.weight_mu.shape, minval=-mu_range, maxval=mu_range)
#         self.weight_sigma = tf.fill(self.weight_sigma.shape, self.std_init / math.sqrt(self.in_features))
#
#         self.bias_mu = tf.keras.backend.random_uniform(self.bias_mu.shape, minval=-mu_range, maxval=mu_range)
#         self.bias_sigma = tf.fill(self.bias_sigma.shape, self.std_init / math.sqrt(self.in_features))
#
#     def reset_noise(self):
#         """Make new noise."""
#         p = tf.random.normal([self.in_features,1])
#         q = tf.random.normal([1,self.out_features])
#         f_p = f(p)
#         f_q = f(q)
#         # outer product
#         self.weight_epsilon = f_p * f_q
#         self.bias_epsilon = tf.squeeze(f_q)
#
#     def call(self, input):
#         """Forward method implementation.
#
#         We don't use separate statements on train / eval mode.
#         It doesn't show remarkable difference of performance.
#         """
#         w = self.weight_mu + self.weight_sigma * self.weight_epsilon
#         b = self.bias_mu + self.bias_sigma * self.bias_epsilon
#
#         return tf.matmul(input, w) + b
#
#     # @staticmethod
#     # def scale_noise(size: int):
#     #     """Set scale to make noise (factorized gaussian noise)."""
#     #     x = tf.random.normal((size,), mean = 0, stddev = 1)
#     #     return tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))
#     @staticmethod
#     def f(x):
#         return tf.multiply(tf.sign(x),tf.pow(tf.abs(x),0.5))
class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, out_features: int, in_features, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(in_features, 0.5),
                                                maxval=1 * 1 / np.power(in_features, 0.5))
        sigma_init = tf.constant_initializer(std_init / np.power(in_features, 0.5))

        self.weight_mu = self.add_weight(name='weight_mu', shape=[self.in_features, self.out_features], trainable=True,
                                         initializer=mu_init)
        self.weight_sigma = self.add_weight(name='weight_sigma', shape=[self.in_features, self.out_features],
                                            trainable=True, initializer=sigma_init)
        self.weight_epsilon = self.add_weight(name='weight_epsilon', shape=[self.in_features, self.out_features])

        self.bias_mu = self.add_weight(name='bias_mu', shape=[self.out_features], trainable=True, initializer=mu_init)
        self.bias_sigma = self.add_weight(name='bias_sigma', shape=[self.out_features], trainable=True,
                                          initializer=sigma_init)
        self.bias_epsilon = self.add_weight(name='bias_epsilon', shape=[self.out_features])

        self.reset_noise()

    def reset_noise(self):
        """Make new noise."""
        p = tf.random.normal([self.in_features, 1])
        q = tf.random.normal([1, self.out_features])
        f_p = self.f(p)
        f_q = self.f(q)
        # outer product
        self.weight_epsilon = f_p * f_q
        self.bias_epsilon = tf.squeeze(f_q)

    @staticmethod
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

    def call(self, input):
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon
        b = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return tf.matmul(input, w) + b


class Network(tf.keras.Model):
    def __init__(self, learning_rate):
        super(Network, self).__init__(name='')
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = NoisyLinear(128, in_features=128)
        self.advantage_output = tf.keras.layers.Dense(action_size)
        self.value_out = tf.keras.layers.Dense(1)
        self.norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))

        self.build((None, state_size))
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                     loss='mse',
                     metrics=['accuracy'])

    def call(self, input_tensor):
        x = self.fc1(input_tensor)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)

        y = self.advantage_output(x)
        y = self.norm_advantage_output(y)
        z = self.value_out(x)
        x = y + z
        return x

    def reset_noise(self):
        """Reset all noisy layers."""

        self.fc2.reset_noise()


class DQNAgent:
    def __init__(self):
        # other hyperparameters
        self.save_graph = True
        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.save_model = True
        self.load_model = False
        self.random = False
        self.dueling = True

        # check the hyperparameters
        if self.random == True:
            self.play = False
            self.isTraining = False
        if self.play == True:
            self.render = True
            self.save_model = False
            self.load_model = True
            self.isTraining = False
            self.keepTraining = False
        if self.keepTraining == True:
            self.load_model = True
        # fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 100
        self.target_network_counter = 0

        # n-step learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # experience replay used SumTree
        # combine agent and PER
        self.batch_size = 32
        self.gamma = 0.9
        self.replay_start_size = 320
        self.experience_replay = SumTree(memory_size)
        self.PER_e = 0.01  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
        self.PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a
        self.PER_b = 0.4
        self.PER_b_increment = 0.002
        self.absolute_error_upper = 1.  # clipped error
        self.experience_number = 0
        # initially, p1=1 total_priority=1,so P(1)=1,w1=batchsize**beta

        if self.load_model:
            self.model = keras.models.load_model('cartpole_dddqn_per_model.h5')
            self.target_model = keras.models.load_model('cartpole_dddqn_per_model.h5')
        else:
            self.model = Network(self.learning_rate)
            self.target_model = Network(self.learning_rate)

    # n-step learning, get the truncated n-step return
    def get_n_step_info(self, n_step_buffer, gamma):
        """Return n step reward, next state, and done."""
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    # these three methods:sample,store,batch_update are used in experience replay
    def sample(self, n):
        mini_batch = []
        batch_index = np.empty((n,), dtype=int)
        batch_ISWeights = np.empty((n,), dtype=float)
        priority_segment = self.experience_replay.total_priority / n
        if self.PER_b < 1:
            self.PER_b += self.PER_b_increment

        min_priority_probability = np.min(
            self.experience_replay.tree[-self.experience_replay.capacity:]) / self.experience_replay.total_priority
        if min_priority_probability == 0:
            min_priority_probability = 1 / memory_size
        # max_weight = (min_priority_probability * memory_size) ** (-self.PER_b)
        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.experience_replay.get_leaf(value)
            sampling_probability = priority / self.experience_replay.total_priority
            # batch_ISWeights[i] = np.power(sampling_probability*memory_size,-self.PER_b) / max_weight
            batch_ISWeights[i] = np.power(sampling_probability / min_priority_probability, -self.PER_b)
            batch_index[i] = index
            mini_batch.append(data)
        return batch_index, mini_batch, batch_ISWeights

    # newly transitions have max_priority or 1 at first transition
    def store(self, experience):
        max_priority = np.max(self.experience_replay.tree[-self.experience_replay.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        if self.experience_number < memory_size:
            self.experience_number += 1

        # n_step
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
            state, action = self.n_step_buffer[0][:2]
            self.experience_replay.add(max_priority, (state, action, reward, next_state, done))

    def batch_update(self, tree_index, abs_errors):
        abs_errors = tf.add(abs_errors, self.PER_e)
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.PER_a)
        for index, priority in zip(tree_index, priorities):
            self.experience_replay.update(index, priority)

    def training(self):
        if self.experience_number >= self.replay_start_size:
            batch_index, batches, batch_ISWeights = self.sample(self.batch_size)
            absolute_errors = []
            buffer_state = [data[0] for data in batches]
            buffer_action = [data[1] for data in batches]
            buffer_reward = [data[2] for data in batches]
            buffer_next_state = [data[3] for data in batches]
            buffer_done = [data[4] for data in batches]

            buffer_state = np.reshape(buffer_state, (self.batch_size, state_size))
            buffer_next_state = np.reshape(buffer_next_state, (self.batch_size, state_size))
            y = self.model(buffer_state).numpy()
            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.model(buffer_next_state).numpy(), axis=1)
            target_y = self.target_model(buffer_next_state).numpy()

            # n_step learning: gamma is also truncated
            # now the experience actually store n-step info
            # such as state[0], action[0], n-step reward, next_state[2] and done[2]
            n_gamma = self.gamma ** self.n_step
            target_network_q_value = target_y[np.arange(self.batch_size), max_action_next]
            # now the experience actually store n-step info
            # such as state[0], action[0], n-step reward, next_state[2] and done[2]
            q_values_req = np.where(buffer_done, buffer_reward, buffer_reward + n_gamma * target_network_q_value)
            absolute_errors = tf.abs(y[np.arange(self.batch_size), buffer_action] - q_values_req)
            y[np.arange(self.batch_size), buffer_action] = q_values_req

            history = self.model.fit(buffer_state, y, batch_size=self.batch_size, epochs=64, verbose=0,
                                     sample_weight=batch_ISWeights)

            self.batch_update(batch_index, absolute_errors)
            self.model.reset_noise()
            self.target_model.reset_noise()
            return history

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            # print('weights updated')
        random_number = np.random.sample()
        action = np.argmax(self.model(state).numpy()[0])
        return action

    def draw(self, rewards, location):
        plt.plot(rewards)
        plt.title('score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Last Score')
        plt.ylim(bottom=0)
        plt.savefig(location)
        plt.close()


agent = DQNAgent()

if agent.isTraining:
    scores_window = deque(maxlen=100)
    start = time.time()
    for episode in range(1, episodes + 1):
        rewards = 0
        state = env.reset()
        state = np.array([state])
        while True:
            action = agent.acting(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = next_state[None, :]
            reward = -10 if done else reward

            agent.store((state, action, reward, next_state, done))
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                scores_window.append(rewards)
                history = agent.training()

                break
        print('\rEpisode {}\tAverage Score: {:.2f}\tper_beta: {:.2f}'.format(episode, np.mean(scores_window),
                                                                             agent.PER_b), end="")

        if np.mean(scores_window) > 195:
            print("\nproblem solved in {} episode with {:.2f} seconds".format(episode, time.time() - start))
            agent.model.save('cartpole_dddqn_per_model.h5')
            agent.draw(episode_rewards, "test.png")
            break
        if episode % 100 == 0:
            print("\nRunning for {:.2f} seconds".format(time.time() - start))
    env.close()
