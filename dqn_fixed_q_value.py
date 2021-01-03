import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import random
import time
from collections import deque
#from memory_profiler import profile
#import tensorflow.keras.backend as K

start_time = time.time()

episodes = 1500
episode_rewards=[]
average_rewards = []
max_reward = 0
max_average_reward = 0
step_limit = 500
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
template = 'episode: {}, rewards: {:.2f}, max reward: {}, mean_rewards: {}, epsilon: {}'

os.environ['CUDA_VISIBLE_DEVICES']='0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpus) > 0
tf.config.experimental.set_memory_growth(gpus[0], True)

class DQNAgent:
    def __init__(self):
        #other hyperparameters
        self.save_graph = True
        self.isTraining = True
        self.keepTraining = False
        self.play = False
        self.render = False
        self.save_model = True
        self.load_model = False
        self.random = False

        #epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 50000
        self.decay_rate = 0.995

        #check the hyperparameters
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
        #fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 1000
        self.target_network_counter = 0
        if self.load_model:
            self.model = keras.models.load_model('cartpole_dqn_mode_4_epoches64.h5')
            self.target_model = keras.models.load_model('cartpole_dqn_mode_4_epoches64.h5')
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

        #experience replay
        self.experience_replay = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.9
        self.replay_start_size = 320

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(20, activation='relu', input_dim=state_size),
            keras.layers.Dense(action_size, activation='elu')]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])
        return model

    # def training(self, model):
    #     if len(self.experience_replay) >= self.replay_start_size:
    #         #if self.epsilon > self.min_epsilon:
    #         #    self.epsilon = self.epsilon * self.decay_rate
    #         batches = np.random.choice(len(self.experience_replay), self.batch_size)
    #         for i in batches:
    #             buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done = self.experience_replay[i]
    #             if buffer_done:
    #                 y_reward = buffer_reward
    #             else:
    #                 y_reward = buffer_reward + self.gamma*np.max(self.target_model.predict(buffer_next_state)[0])
    #             #only one output, which has the shape(1,2)
    #             y = model.predict(buffer_state)
    #             y[0][buffer_action] = y_reward
    #             model.fit(buffer_state, y, epochs=1, verbose=0)

    def training(self):
        if len(self.experience_replay) >= self.replay_start_size:
            #if self.epsilon > self.min_epsilon:
            #    self.epsilon = self.epsilon * self.decay_rate
            batches = random.sample(self.experience_replay, self.batch_size)
            buffer_state = [data[0] for data in batches]
            buffer_action = [data[1] for data in batches]
            buffer_reward = [data[2] for data in batches]
            buffer_next_state = [data[3] for data in batches]
            buffer_done = [data[4] for data in batches]

            buffer_state = np.reshape(buffer_state,(self.batch_size,state_size))
            buffer_next_state = np.reshape(buffer_next_state,(self.batch_size,state_size))
            y = self.model.predict(buffer_state)
            target_y = self.target_model.predict(buffer_next_state)
            for i in range(0,self.batch_size):
                done = buffer_done[i]
                if done:
                    y_reward = buffer_reward[i]
                else:
                    y_reward = buffer_reward[i] + self.gamma * np.max(target_y[i])
                #only one output, which has the shape(1,2)
                y[i][buffer_action[i]] = y_reward
            self.model.fit(buffer_state, y, epochs=64, verbose=0)

    def acting(self,state):
        if self.render:
            env.render()
        self.target_network_counter += 1
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            #print('weights updated')
        random_number = np.random.sample()
        if random_number > self.epsilon:
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = np.random.randint(action_size)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.linear_annealed
        return action

    #Since heavily importing into the global namespace may result in unexpected behavior,
    #the use of pylab is strongly discouraged
    def draw(self,episode,episode_rewards,average_rewards,max_reward,max_average_reward,location1,location2):
        plt.figure()
        plt.plot(range(episode+1), episode_rewards, 'b')
        plt.title('score with episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.ylim(0.0,max_reward+10)
        plt.savefig(location1)
        #average_rewards
        plt.figure()
        plt.title('mean_score with episodes')
        plt.xlabel('mean_episodes')
        plt.ylabel('score')
        plt.ylim(0.0,max_average_reward+10)
        plt.plot(range(episode+1),average_rewards,'r')
        plt.savefig(location2)

agent = DQNAgent()
if agent.isTraining:
    for episode in range(episodes):
        rewards = 0
        state = env.reset()
        state = np.reshape(state,[1,4])
        while True:
            action = agent.acting(state)
            next_state, reward, done, _= env.step(action)
            rewards += reward
            next_state = np.reshape(next_state,[1,4])
            reward = -100 if done else reward
            agent.experience_replay.append((state,action,reward,next_state,done))
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_average_reward = max(max_average_reward,average_reward)
                max_reward = max(max_reward,rewards)
                agent.training()
                break
        if (episode + 1) % 50==0:
            print(template.format(episode,rewards,max_reward,average_reward,agent.epsilon))
            if agent.save_model:
                agent.model.save('cartpole_dqn_model.h5')
                print('model saved')
            if agent.save_graph:
                agent.draw(episode,episode_rewards,average_rewards,max_reward,max_average_reward,"./dqn_training_cartpole.png","./dqn_training_cartpole_ave.png")
        if (episode + 1) % 100 == 0:
            end_time = time.time()
            print('running time: {:.2f} minutes'.format((end_time-start_time) / 60))
            print('average score in last ten episodes is: {}'.format(tf.reduce_mean(episode_rewards[-10:])))
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
            #env.render()
            action = np.random.randint(action_size)
            next_state, reward, done, _= env.step(action)
            rewards += reward
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward,rewards)
                max_average_reward = max(max_average_reward,average_reward)
                print(template.format(episode,rewards,max_reward,average_reward,"Not used"))
                break
    agent.draw(episode,episode_rewards,average_rewards,max_reward,max_average_reward,"./dqn_random_cartpole.png","./dqn_random_cartpole_ave.png")


if agent.play:
    episode_rewards = []
    average_rewards = []
    max_average_reward = 0
    max_reward = 0
    for episode in range(10):
        state = env.reset()
        rewards = 0
        state = np.reshape(state,[1,4])
        while True:
            env.render()
            action = np.argmax(agent.model.predict(state)[0])
            next_state, reward, done, _= env.step(action)
            rewards += reward
            next_state = np.reshape(next_state,[1,4])
            state = next_state
            if done or rewards >= step_limit:
                episode_rewards.append(rewards)
                average_reward = tf.reduce_mean(episode_rewards).numpy()
                average_rewards.append(average_reward)
                max_reward = max(max_reward,rewards)
                max_average_reward = max(max_average_reward,average_reward)
                print(template.format(episode,rewards,max_reward,average_reward,"Not used"))
                break
    agent.draw(episode,episode_rewards,average_rewards,max_reward,max_average_reward,"./dqn_playing_cartpole.png","./dqn_playing_cartpole_ave.png")
