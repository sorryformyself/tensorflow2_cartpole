import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import time
from collections import deque

tf.get_logger().setLevel('ERROR')
start_time = time.time()
# solved in 381 episode
episodes = 1500
episode_rewards = []
average_rewards = []
last_average_rewards = []
max_reward = 0
max_average_reward = 0
step_limit = 200
memory_size = 100000
env = gym.make('CartPole-v1')
env.seed(777)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
template = 'episode: {}, rewards: {:.2f}, max reward: {}, mean_rewards: {}, epsilon: {}'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpus) > 0
tf.config.experimental.set_memory_growth(gpus[0], True)


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

        if self.load_model:
            self.model = keras.models.load_model('cartpole_dddqn_per_model.h5')
            self.target_model = keras.models.load_model('cartpole_dddqn_per_model.h5')
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

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
        self.experience_replay.add(max_priority, experience)

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
            buffer_state = [data[0] for data in batches]
            buffer_action = [data[1] for data in batches]
            buffer_reward = [data[2] for data in batches]
            buffer_next_state = [data[3] for data in batches]
            buffer_done = [data[4] for data in batches]

            buffer_state = np.reshape(buffer_state, (self.batch_size, state_size))
            buffer_next_state = np.reshape(buffer_next_state, (self.batch_size, state_size))
            y = self.model.predict(buffer_state)
            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.model.predict(buffer_next_state), axis=1)
            target_y = self.target_model.predict(buffer_next_state)
            for i in range(0, self.batch_size):
                done = buffer_done[i]
                if done:
                    y_reward = buffer_reward[i]
                else:
                    # then calculate the q-value in target network
                    target_network_q_value = target_y[i, max_action_next[i]]
                    y_reward = buffer_reward[i] + self.gamma * target_network_q_value
                # only one output, which has the shape(1,2)
                # prediction value - actual value
                # the value between implemented action and maximum action
                absolute_errors.append(tf.abs(y[i, buffer_action[i]] - y_reward))
                y[i, buffer_action[i]] = y_reward
            history = self.model.fit(buffer_state, y, batch_size=self.batch_size, epochs=64, verbose=0,
                                     sample_weight=batch_ISWeights)

            self.batch_update(batch_index, absolute_errors)
            return history

    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            # print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        return action

    # Since heavily importing into the global namespace may result in unexpected behavior,
    # the use of pylab is strongly discouraged
    def draw(self, episode, episode_rewards, average_rewards, location):
        plt.figure(figsize=(15, 6))
        plt.subplots_adjust(wspace=0.3)
        plt.subplot(1, 2, 1)
        # using polynomial to fit
        p1 = np.poly1d(np.polyfit(range(episode + 1), episode_rewards, 3))
        yvals = p1(range(episode + 1))
        # plt.plot(range(episode+1), yvals, 'b')
        plt.plot(range(episode + 1), episode_rewards, 'b')
        plt.title('score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.ylim(bottom=0)
        # average_rewards
        plt.subplot(1, 2, 2)
        plt.plot(range(episode + 1), average_rewards, 'r')
        plt.title('mean_score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.ylim(bottom=0)
        plt.savefig(location)
        plt.close()

    def redraw(self, rewards, location):
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
    for episode in range(episodes):
        rewards = 0
        state = env.reset()
        state = np.array([state])
        while True:
            action = agent.acting(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = next_state[None, :]
            reward = -10 if done else reward
            # agent.experience_replay.append((state,action,reward,next_state,done))
            agent.store((state, action, reward, next_state, done))
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                scores_window.append(rewards)
                # max_average_reward = max(max_average_reward,average_reward)
                max_reward = max(max_reward, rewards)
                history = agent.training()

                break
        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tepsilon:{:.2f}\tbeta: {:.2f}'.format(episode, np.mean(scores_window),
                                                                                       agent.epsilon, agent.PER_b),
            end="")

        if np.mean(scores_window) > 195:
            print("\nproblem solved in {} episode in {}".format(episode, time.time() - start))
            agent.model.save('cartpole_dddqn_per_model.h5')
            agent.redraw(episode_rewards, "test.png")
            break
        if episode % 100 == 0:
            print("100 episodes {}".format(time.time() - start))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

        # if (episode + 1) % 50==0:
        #     print(template.format(episode,rewards,max_reward,average_reward,agent.epsilon))
        #     plt.xlabel("epoches")
        #     plt.ylabel("loss")
        #     plt.ylim(bottom=0)
        #     plt.plot(history.history['accuracy'])
        #     plt.show()
        #     if agent.save_model:
        #         agent.model.save('cartpole_dddqn_per_model.h5')
        #         print('model saved')
        #     if agent.save_graph:
        #         last_average_rewards.append(tf.reduce_mean(episode_rewards[-50:]))
        #         agent.redraw(episode,last_average_rewards,"test.png")
        #         agent.draw(episode,episode_rewards,average_rewards,"./dddqn_per_training_cartpole.png")
        # if (episode + 1) % 100 == 0:
        #     end_time = time.time()
        #
        #     print('running time: {:.2f} minutes'.format((end_time-start_time) / 60))
        #     print('average score in last 100 episodes is: {}'.format(last_average_rewards[-1]))
    env.close()

if agent.random:
    episode_rewards = []
    average_rewards = []
    max_average_reward = 0
    max_reward = 0
    for episode in range(3000):
        state = env.reset()
        rewards = 0
        while True:
            env.render()
            action = np.random.randint(action_size)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward, rewards)
                print(template.format(episode, rewards, max_reward, average_reward, "Not used"))
                break
    agent.draw(episode, episode_rewards, average_rewards, "./random_cartpole.png")

if agent.play:
    episode_rewards = []
    average_rewards = []
    max_average_reward = 0
    max_reward = 0
    for episode in range(10):
        state = env.reset()
        rewards = 0
        state = np.reshape(state, [1, 4])
        while True:
            env.render()
            action = np.argmax(agent.model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            next_state = np.reshape(next_state, [1, 4])
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward, rewards)
                # max_average_reward = max(max_average_reward,average_reward)
                print(template.format(episode, rewards, max_reward, average_reward, "Not used"))
                break
    agent.draw(episode, episode_rewards, average_rewards, "./dddqn_per_playing_cartpole.png")
