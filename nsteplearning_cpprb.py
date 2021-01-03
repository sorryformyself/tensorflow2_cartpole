import os
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from cpprb import PrioritizedReplayBuffer
from gym.envs.mspacman_array_state.Utils import Utils

tf.get_logger().setLevel('ERROR')
episodes = 1000
episode_rewards = []
step_limit = 200
memory_size = 100000
env = gym.make('CartPole-v1')
env.seed(777)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

saveFileName = 'cartpole_nstep'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpus) > 0
# tf.config.experimental.set_memory_growth(gpus[0], True)


# Episode 141	Average Score: 195.54	epsilon:0.01	per_beta: 1.00
# problem solved in 141 episode with 22.88 seconds
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
        if self.random:
            self.play = False
            self.isTraining = False
        if self.play:
            self.render = True
            self.save_model = False
            self.load_model = True
            self.isTraining = False
            self.keepTraining = False
        if self.keepTraining:
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
        self.PER_e = 0.01  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
        self.PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a
        self.PER_b = 0.4
        self.PER_b_increment = 0.005
        self.absolute_error_upper = 1.  # clipped error
        self.experience_number = 0

        env_dict = {"obs": {"shape": (state_size,)},
                    "act": {},
                    "rew": {},
                    "next_obs": {"shape": (state_size,)},
                    "done": {}}
        self.experience_replay = PrioritizedReplayBuffer(memory_size, env_dict=env_dict, alpha=self.PER_a,
                                                         eps=self.PER_e)

        # initially, p1=1 total_priority=1,so P(1)=1,w1=batchsize**beta

        if self.load_model:
            self.model = keras.models.load_model('cartpole_nstep.h5')
            self.target_model = keras.models.load_model('cartpole_nstep.h5')
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

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

    # newly transitions have max_priority or 1 at first transition
    def store(self, experience):
        if self.experience_number < memory_size:
            self.experience_number += 1

        # n_step
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
            state, action = self.n_step_buffer[0][:2]
            self.experience_replay.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

    def batch_update(self, tree_index, abs_errors):
        abs_errors = tf.add(abs_errors, self.PER_e)
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.PER_a)
        self.experience_replay.update_priorities(tree_index, priorities)

    # DDDQN dueling double DQN, the network structure should change
    def create_model(self):
        inputs = tf.keras.Input(shape=(state_size,))
        fc1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        fc3 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc2)
        if self.dueling:
            value_out = tf.keras.layers.Dense(1, activation='linear')(fc3)
            norm_advantage_output = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
            # outputs = tf.keras.layers.Add()([value_out,advantage_output-tf.reduce_mean(advantage_output,axis=1,keepdims=True)])
            outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
            model = tf.keras.Model(inputs, outputs)
        else:
            model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.summary()
        return model

    # @profile
    # @Utils.timer()
    def training(self):
        if self.experience_number >= self.replay_start_size:
            s = self.experience_replay.sample(self.batch_size, beta=self.PER_b)
            # if self.PER_b < 1:
            #     self.PER_b += self.PER_b_increment
            buffer_state = s['obs']
            buffer_action = np.squeeze(s['act']).astype(np.int)
            buffer_reward = np.squeeze(s['rew'])
            buffer_next_state = s['next_obs']
            buffer_done = np.squeeze(s['done'])

            # n_step learning: gamma is also truncated
            n_gamma = self.gamma ** self.n_step

            y = self.local_inference(buffer_state).numpy()
            # DDQN double DQN: choose action first in current network,
            # no axis=1 will only have one value
            max_action_next = np.argmax(self.local_inference(buffer_next_state).numpy(), axis=1)
            target_y = self.target_inference(buffer_next_state).numpy()
            target_network_q_value = target_y[np.arange(self.batch_size), max_action_next]
            # now the experience actually store n-step info
            # such as state[0], action[0], n-step reward, next_state[2] and done[2]
            q_values_req = np.where(buffer_done, buffer_reward,
                                    buffer_reward + n_gamma * target_network_q_value).astype(np.float32)
            absolute_errors = tf.abs(y[np.arange(self.batch_size), buffer_action] - q_values_req)
            y[np.arange(self.batch_size), buffer_action] = q_values_req
            history = self.model.fit(buffer_state, y, batch_size=self.batch_size, epochs=64, verbose=0,
                                     sample_weight=s['weights'])
            self.batch_update(s['indexes'], absolute_errors)
            return history

    # tf function train_batch 78.69s 128score
    # train_batch
    @tf.function
    def local_inference(self, x):
        return self.model(x)

    @tf.function
    def target_inference(self, x):
        return self.target_model(x)

    @tf.function
    def train_batch(self, x, y, sample_weight):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss(y, predictions, sample_weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.weights, local_model.weights):
            target_param.assign(tau * local_param + (1.0 - tau) * target_param)

    # 220         1       5318.0   5318.0     98.1
    def acting(self, state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        # self.soft_update(self.model, self.target_model, 0.001)
        # print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.local_inference(state).numpy()[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        return action


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
        Utils.printBeta(episode, scores_window, beta=agent.PER_b, epsilon=agent.epsilon,
                        steps=agent.target_network_counter)
        if np.mean(scores_window) > 195:
            Utils.printSolvedTime(start, episode)
            agent.model.save(saveFileName + '.h5')
            Utils.saveRewards(saveFileName, episode_rewards)
            Utils.saveThreePlots(saveFileName, episode_rewards)
            break
        if episode % 100 == 0:
            Utils.printRunningTime(start, agent.target_network_counter)
    env.close()
