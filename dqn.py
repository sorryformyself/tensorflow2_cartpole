import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import pylab
from collections import deque

episodes = 2000
episode_rewards = []
env = gym.make('CartPole-v0')
min_epsilon = 0.1
epsilon = 1
decay_rate = 0.995
max_reward = 0
experience_replay = deque(maxlen=100000)
batch_size = 64
gamma = 0.9
learning_rate = 0.0001
save_graph = True
fixed_q_value_steps = 200
current_steps = 0
template = 'episode: {}, rewards: {:.2f}, max reward: {}, mean_rewards: {}, epsilon: {}'
from gym.envs.mspacman_array_state.Utils import Utils


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(4,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='relu')]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mse',
                  metrics=['accuracy'])
    return model


model = create_model()
target_model = create_model()


# Total time: 10.6749 s
# File: D:/OneDrive - bupt.edu.cn/tensorflow2-RL-demo/cartpole/dqn.py
# Function: training at line 42
#
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     42                                           @Utils.lp_wrapper()
#     43                                           def training(model):
#     44                                               # training
#     45         1         19.0     19.0      0.0      if len(experience_replay) >= batch_size:
#     46         1       1038.0   1038.0      0.0          batches = np.random.choice(len(experience_replay), batch_size)
#     47        65       2654.0     40.8      0.0          for i in batches:
#     48        64       1587.0     24.8      0.0              buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done = experience_replay[i]
#     49        64        376.0      5.9      0.0              if buffer_done:
#     50         3         19.0      6.3      0.0                  y_reward = buffer_reward
#     51                                                       else:
#     52        61   30212834.0 495292.4     28.3                  y_reward = buffer_reward + gamma * np.max(target_model.predict(buffer_next_state)[0])
#     53                                                       # only one output, which has the shape(1,2)
#     54        64   33628409.0 525443.9     31.5              y = model.predict(buffer_state)
#     55        64       4577.0     71.5      0.0              y[0][buffer_action] = y_reward
#     56        64   42897394.0 670271.8     40.2              model.fit(buffer_state, y, epochs=1, verbose=0)

@Utils.lp_wrapper()
def training(model):
    # training
    if len(experience_replay) >= batch_size:
        batches = np.random.choice(len(experience_replay), batch_size)
        for i in batches:
            buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done = experience_replay[i]
            if buffer_done:
                y_reward = buffer_reward
            else:
                y_reward = buffer_reward + gamma * np.max(target_model.predict(buffer_next_state)[0])
            # only one output, which has the shape(1,2)
            y = model.predict(buffer_state)
            y[0][buffer_action] = y_reward
            model.fit(buffer_state, y, epochs=1, verbose=0)


for episode in range(episodes):
    if epsilon > min_epsilon:
        epsilon = epsilon * decay_rate
    rewards = 0
    state = env.reset()
    state = np.reshape(state, [1, 4])
    while True:
        # env.render()
        current_steps += 1
        if current_steps % fixed_q_value_steps == 0:
            target_model.set_weights(model.get_weights())
            print('weights updated')
        random_number = np.random.sample()
        if random_number > epsilon:
            action = np.argmax(model.predict(state)[0])
        else:
            action = np.random.randint(2)
        next_state, reward, done, _ = env.step(action)
        # state, reward, done, _= env.step(np.random.randint(2))
        rewards += reward
        next_state = np.reshape(next_state, [1, 4])
        reward = -100 if done else reward
        experience_replay.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            episode_rewards.append(rewards)
            max_reward = max(max_reward, rewards)
            print(template.format(episode, rewards, max_reward, tf.reduce_mean(episode_rewards).numpy(), epsilon))
            training(model)
            break
    if (episode + 1) % 50 == 0:
        model.save('dqn_model.h5')
        print('model saved')
        if save_graph:
            pylab.plot(range(episode + 1), episode_rewards, 'b')
            pylab.savefig("./dqn_results_fixed.png")
env.close()
