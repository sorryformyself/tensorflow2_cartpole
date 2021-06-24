import os
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from cpprb import PrioritizedReplayBuffer

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

    def store(self, experience):
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
            state, action = self.n_step_buffer[0][:2]
            self.experience_replay.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

    def create_model(self):
        inputs = tf.keras.Input(shape=(state_size,))
        fc1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc2)

        value_out = tf.keras.layers.Dense(1, activation='linear')(fc2)
        norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
        outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        if self.experience_replay.get_stored_size() > self.batch_size:
            samples = self.experience_replay.sample(self.batch_size)
            td_errors, loss = self._train_body(samples)
            self.experience_replay.update_priorities(
                samples["indexes"], td_errors.numpy() + 1e-6)

    @tf.function
    def _train_body(self, samples):
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error_body(samples["obs"], samples["act"], samples["rew"],
                                                    samples["next_obs"], samples["done"])
            loss = tf.reduce_mean(tf.square(td_errors))  # huber loss seems no use
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return td_errors, loss

    @tf.function
    def _compute_td_error_body(self, states, actions, rewards, next_states, dones):
        rewards = tf.cast(tf.squeeze(rewards), dtype=tf.float32)
        dones = tf.cast(tf.squeeze(dones), dtype=tf.bool)
        actions = tf.cast(actions, dtype=tf.int32)  # (batch_size, 1)
        batch_size_range = tf.expand_dims(tf.range(self.batch_size), axis=1)  # (batch_size, 1)

        # get current q value
        current_q_indexes = tf.concat(values=(batch_size_range, actions), axis=1)  # (batch_size, 2)
        current_q = tf.gather_nd(self.model(states), current_q_indexes)  # (batch_size, )

        # get target q value using double dqn
        max_next_q_indexes = tf.argmax(self.model(next_states), axis=1, output_type=tf.int32)  # (batch_size, )
        indexes = tf.concat(values=(batch_size_range,
                                    tf.expand_dims(max_next_q_indexes, axis=1)), axis=1)  # (batch_size, 2)
        target_q = tf.gather_nd(self.target_model(next_states), indexes)  # (batch_size, )

        target_q = tf.where(dones, rewards, rewards + self.gamma * target_q)  # (batch_size, )
        # don't want change the weights of target network in backpropagation, so tf.stop_gradient()
        # but seems no use
        td_errors = tf.abs(current_q - tf.stop_gradient(target_q))
        return td_errors

    def select_action(self, state):
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.epsilon = max(self.epsilon - self.linear_annealed, self.min_epsilon)
        if np.random.sample() <= self.epsilon:
            return np.random.randint(action_size)
        return self._get_action_body(state).numpy()

    @tf.function
    def _get_action_body(self, state):
        state = tf.expand_dims(state, axis=0)
        qvalues = self.model(state)[0]
        return tf.argmax(qvalues)


agent = DQNAgent()

if agent.isTraining:
    scores_window = deque(maxlen=100)
    start = time.time()
    for episode in range(1, episodes + 1):
        rewards = 0
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            reward = -10 if done else reward

            agent.store((state, action, reward, next_state, done))
            state = next_state
            agent.train()
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                scores_window.append(rewards)
                break

        print('\rEpisode {}\tAverage Score: {:.2f}'
              '\tepsilon:{:.2f}\tper_beta: {:.2f}\tstep: {}'.format(episode, np.mean(scores_window), agent.epsilon,
                                                                    agent.PER_b, agent.target_network_counter), end='')

        if np.mean(scores_window) > 195:
            m, s = divmod(time.time() - start, 60)
            h, m = divmod(m, 60)
            print("\nproblem solved in episode %d with %d hours %d minutes %d seconds" % (episode, h, m, s))
            agent.model.save(saveFileName + '.h5')
            # Utils.saveRewards(saveFileName, episode_rewards)
            # Utils.saveThreePlots(saveFileName, episode_rewards)
            break
        if episode % 100 == 0:
            m, s = divmod(time.time() - start, 60)
            h, m = divmod(m, 60)
            print("\nRunning for %d steps with %d hours %d minutes %d seconds" % (agent.target_network_counter, h, m, s))
    env.close()
