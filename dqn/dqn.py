import tensorflow as tf
import gym
import numpy as np
import tensorflow.keras as keras
import pylab
from collections import deque

episodes = 2000
episode_rewards = []
env = gym.make('CartPole-v1')
min_epsilon = 0.01
epsilon = 1
decay_rate = 0.995
epsilon_decay = (1 - min_epsilon) / 2000
max_reward = 0
experience_replay = deque(maxlen=20000)
batch_size = 32
gamma = 0.9
learning_rate = 0.0001
save_graph = True
fixed_q_value_steps = 100
current_steps = 0
template = 'episode: {}, rewards: {:.2f}, max reward: {}, mean_rewards: {:.2f}, epsilon: {:.2f}'


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(4,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='relu')]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mse',
                  metrics=['accuracy'])
    return model


model = create_model()
target_model = create_model()


def training(model):
    # training
    if len(experience_replay) >= batch_size:
        batches = np.random.choice(len(experience_replay), batch_size)
        experiences = [experience_replay[i] for i in batches]

        buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done = zip(*experiences)

        buffer_state = np.asarray(buffer_state).squeeze()
        buffer_action = np.asarray(buffer_action).squeeze()
        buffer_reward = np.asarray(buffer_reward).squeeze()
        buffer_next_state = np.asarray(buffer_next_state).squeeze()
        buffer_done = np.asarray(buffer_done).squeeze()

        y = model(buffer_state).numpy()
        y_reward = np.where(buffer_done, buffer_reward,
                            buffer_reward + gamma * np.max(target_model(buffer_next_state).numpy()))

        y[np.arange(batch_size), buffer_action] = y_reward
        model.fit(buffer_state, y, epochs=32, verbose=0)


for episode in range(episodes):

    rewards = 0
    state = env.reset()
    state = np.reshape(state, [1, 4])
    while True:
        # env.render()

        current_steps += 1
        if current_steps % fixed_q_value_steps == 0:
            target_model.set_weights(model.get_weights())
        random_number = np.random.sample()
        if random_number > epsilon:
            action = np.argmax(model(state).numpy()[0])
        else:
            action = np.random.randint(2)
        if epsilon > min_epsilon:
            epsilon = epsilon - epsilon_decay
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        next_state = np.reshape(next_state, [1, 4])
        reward = -10 if done else reward
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
