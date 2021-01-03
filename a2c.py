# a2c

import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
from collections import deque

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    # Sample a random categorical action from the given logits.
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Actor(tf.keras.Model):
    def __init__(self, num_actions, learning_rate=0.001):
        super(Actor, self).__init__(name='actor')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions)
        self.dist = ProbabilityDistribution()
        self.learning_rate = learning_rate

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.logits(x)

    def get_action(self, state):
        logits = self(state)
        action = self.dist(logits)
        return np.squeeze(action, axis=-1)

class Critic(tf.keras.Model):
    def __init__(self, learning_rate=0.001):
        super(Critic, self).__init__(name='critic')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)
        self.learning_rate = learning_rate

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.value(x)

class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma=0.99, value_c=0.5, entropy_c=1e-4, batch_size=64):
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.actor = Actor(action_size, learning_rate)
        self.actor.optimizer = tf.keras.optimizers.Adam(self.actor.learning_rate)

        self.critic = Critic(learning_rate)
        self.critic.optimizer = tf.keras.optimizers.Adam(self.critic.learning_rate)

        self.actions = np.empty((batch_size,), dtype=np.int32)
        self.rewards, self.dones, self.values = np.empty((3, batch_size))
        self.states = np.empty((batch_size,) + env.observation_space.shape)

    def step(self, state, action, next_state, reward, done):
        self.states[i] = state
        self.actions[i] = action
        self.values[i] = self.critic(state[None, :])
        self.rewards[i] = reward
        self.dones[i] = done

    def train(self, state):
        state_value = self.critic(state[None, :])
        returns, advs = self.returns_advantages(self.rewards, self.dones, self.values, state_value)
        acts_and_advs = np.concatenate([self.actions[:, None], advs[:, None]], axis=-1)
        self.actor_train(self.states, acts_and_advs)
        self.critic_train(self.states, returns)

    def actor_train(self, states, acts_and_advs):
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            loss = self.actor_loss(acts_and_advs, logits)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss

    def critic_train(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            loss = self.critic_loss(returns, values)
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    def returns_advantages(self, rewards, dones, values, state_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), np.squeeze(state_value,-1))
        advantages = []
        advantage = 0
        next_value = 0
        trace_decay = 1
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            td_error = r + next_value * self.gamma * (1-d) - v
            advantage = td_error + advantage * self.gamma * trace_decay * (1-d)
            next_value = v
            advantages.insert(0, advantage)
        #backview and forwardview????
        returns = returns[:-1]
        advantages = np.array(advantages)
        # mean,variance = tf.nn.moments(advantages, axes=[0])
        # advantages = (advantages-mean)/variance
        #print(advantages.shape)
        # Advantages are equal to returns - baseline (value estimates in our case).
        # advantages = returns - values
        return returns, advantages
        #gae

#normal
# episode: 1392   Average Score: 195.06
# problem solved in 2084 episode with 368.52 seconds
# gae
# episode: 2632   Average Score: 195.63
# problem solved in 4691 episode with 830.77 seconds
    # y true y pred
    def critic_loss(self, returns, values):
        return self.value_c * tf.keras.losses.MSE(returns, values)

    def actor_loss(self, actions_and_advantages, logits):
        # to equally 2 segments
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        return policy_loss - self.entropy_c * entropy_loss


env = gym.make('CartPole-v0')
action_size = env.action_space.n
state_size = 4
learning_rate = 0.0001
agent = Agent(state_size, action_size, learning_rate)


start = time.time()
scores_window = deque(maxlen=100)
scores_window.append(0.0)
state = env.reset()
episodes = 10000
batch_size = 64
scores_window = deque(maxlen=100)
scores_window.append(0.0)
current = 0
for episode in range(1, episodes+1):
    for i in range(batch_size):
        action = agent.actor.get_action(state[None, :])
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, next_state, reward, done)
        scores_window[-1] += reward
        state = next_state
        if done:
            scores_window.append(0.0)
            state = env.reset()
            current += 1
    agent.train(state)
    print('\repisode: {} \tAverage Score: {:.2f}'.format(current, np.mean(scores_window)), end="")
    if episode % 100 == 0:
        print("\nRunning for {:.2f} seconds".format(time.time()-start))
    if np.mean(scores_window)>195:
        print("\nproblem solved in {} episode with {:.2f} seconds".format(episode, time.time()-start))
        # self.actor.save('cartpole_a2c_actor.h5')
        # self.critic.save('cartpole_a2c_critic.h5')
        # self.draw(episode_rewards,"a2c.png")
        break

#
# #step counter
# T = 0
# t = 1
# while T<Tmax:
#     #reset gradients for actor and Critic
#     #synchronize thread's weights
#     tstart = t
#     state = env.reset()
#     while true:
#         # ProbabilityDistribution
#         action = actor(state)
#         n_s, r, d, _ = env.step(action)
#         t += 1
#         T += 1
#         if terminal or t'-tstart = tmax:
#             R = 0 if terminal else critic(state)
#             #n step
#             for i in range(t-1,tstart+1):
#                 R = reward(i) + gamma*R
#                 #update dtheta actor using advantage function
#                 #update dthetacritic using MSE
#
#             update overall gradient using this dtheta
#             break
