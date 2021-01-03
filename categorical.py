import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
import copy
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

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpus) > 0
# tf.config.experimental.set_memory_growth(gpus[0], True)
saveFileName = 'categorical'


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


class Network(tf.keras.Model):
    def __init__(self, support, atom_size):
        super(Network, self).__init__()

        self.support = support
        self.atom_size = atom_size

        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(128)
        self.advantage_output = tf.keras.layers.Dense(atom_size * action_size)
        self.value_out = tf.keras.layers.Dense(1 * atom_size)
        self.norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.build((None, state_size))

    def call(self, input_tensor):
        dist = self.dist(input_tensor)
        x = tf.reduce_sum(dist * self.support, axis=2)
        return x

    def dist(self, x):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        y = self.advantage_output(x)
        y = self.norm_advantage_output(y)
        y = tf.reshape(y, (-1, action_size, self.atom_size))
        z = self.value_out(x)
        z = tf.reshape(z, (-1, 1, self.atom_size))
        x = y + z
        dist = tf.nn.softmax(x, axis=-1)
        return dist


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
        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 2000
        self.decay_rate = 0.995

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
            self.epsilon = self.min_epsilon
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
        self.batch_size = 64
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

        # categorical DQN
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51
        self.support = np.linspace(
            self.v_min, self.v_max, self.atom_size
        )
        if self.load_model:
            self.model = keras.models.load_model(saveFileName + '.h5')
            self.target_model = keras.models.load_model(saveFileName + '.h5')
        else:
            self.model = Network(self.support, self.atom_size)
            self.target_model = Network(self.support, self.atom_size)
            # self.target_model.predict(np.zeros((1,state_size)))

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

    # DDDQN dueling double DQN, the network structure should change
    def create_model(self):
        inputs = tf.keras.Input(shape=(state_size,))
        fc1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc2)
        if self.dueling:
            value_out = tf.keras.layers.Dense(1, activation='linear')(fc2)
            norm_advantage_output = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
            # outputs = tf.keras.layers.Add()([value_out,advantage_output-tf.reduce_mean(advantage_output,axis=1,keepdims=True)])
            outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
            model = tf.keras.Model(inputs, outputs)
        else:
            model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def training(self):
        if self.experience_number >= self.replay_start_size:
            # if self.epsilon > self.min_epsilon:
            #    self.epsilon = self.epsilon * self.decay_rate
            # batches = random.sample(self.experience_replay, self.batch_size)
            batch_index, batches, batch_ISWeights = self.sample(self.batch_size)
            absolute_errors = []
            buffer_state = np.vstack([data[0] for data in batches])
            buffer_action = np.vstack([data[1] for data in batches])
            buffer_reward = np.vstack([data[2] for data in batches])
            buffer_next_state = np.vstack([data[3] for data in batches])
            buffer_done = np.vstack([data[4] for data in batches])

            # categorical DQN
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.model(buffer_next_state).numpy(), axis=1)
            next_dist = self.target_model.dist(buffer_next_state)
            next_dist = tf.gather_nd(next_dist, [[i, j] for i, j in enumerate(max_action_next)])

            n_gamma = self.gamma ** self.n_step

            t_z = buffer_reward + (1 - buffer_done) * n_gamma * self.support
            t_z = tf.clip_by_value(t_z, self.v_min, self.v_max)
            b = tf.cast((t_z - self.v_min) / delta_z, tf.float32)
            l = tf.cast(tf.math.floor(b), tf.int32)
            u = tf.cast(tf.math.ceil(b), tf.int32)

            offset = tf.tile(tf.cast(tf.linspace(
                0., (self.batch_size - 1.) * self.atom_size, self.batch_size
            ), tf.int32)[:, None]
                             , (1, self.atom_size))
            proj_dist = np.zeros(next_dist.shape, dtype=np.float32)

            loffset = tf.reshape((l + offset), (-1,))
            uoffset = tf.reshape((u + offset), (-1,))
            u_next = tf.reshape((next_dist * (tf.cast(u, tf.float32) - b)), (-1,))
            l_next = tf.reshape((next_dist * (b - tf.cast(l, tf.float32))), (-1,))

            proj_dist = tf.add(tf.reshape(proj_dist, (-1,)), tf.gather(u_next, loffset))
            proj_dist = tf.add(tf.reshape(proj_dist, (-1,)), tf.gather(l_next, uoffset))
            proj_dist = tf.reshape(proj_dist, (self.batch_size, self.atom_size))

            for i in range(64):
                with tf.GradientTape() as tape:
                    dist = self.model.dist(buffer_state)
                    log_p = tf.math.log(tf.gather_nd(dist, [[i, j] for i, j in enumerate(buffer_action.squeeze(-1))]))
                    elementwise_loss = tf.reduce_sum(-(proj_dist * log_p), axis=1)
                    absolute_errors = np.abs(elementwise_loss)
                    loss = tf.reduce_mean(batch_ISWeights.astype(np.float32) * elementwise_loss)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.batch_update(batch_index, absolute_errors)

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1

        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

            # print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.model(state).numpy()[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
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
                agent.training()
                break
        print('\rEpisode {}\tAverage Score: {:.2f}\tepsilon:{:.2f}\tper_beta: {:.2f}'.format(episode,
                                                                                             np.mean(scores_window),
                                                                                             agent.epsilon,
                                                                                             agent.PER_b), end="")

        if np.mean(scores_window) > 195:
            print("\nproblem solved in {} episode with {:.2f} seconds".format(episode, time.time() - start))
            agent.draw(episode_rewards, "test.png")
            agent.model.save(saveFileName + '.h5')
            break
        if episode % 100 == 0:
            print("\nRunning for {:.2f} seconds".format(time.time() - start))
    env.close()
