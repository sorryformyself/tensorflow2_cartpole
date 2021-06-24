import multiprocessing
import time
from collections import deque
from multiprocessing import Process, Event, Value
from multiprocessing.managers import SyncManager

import gym
import numpy as np
import tensorflow as tf
import datetime
import os
from cpprb import PrioritizedReplayBuffer, ReplayBuffer

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
memory_size = 100000

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpus) > 0
tf.config.experimental.set_memory_growth(gpus[0], True)


class Agent:
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

        # experience replay used SumTree
        # combine agent and PER
        self.batch_size = 64
        self.gamma = 0.9
        self.n_warmup = 320

        # epsilon greedy exploration
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.01
        self.linear_annealed = (self.initial_epsilon - self.min_epsilon) / 1000
        self.decay_rate = 0.995
        self.epsilon = max(self.initial_epsilon - self.linear_annealed * self.n_warmup,
                           self.min_epsilon)

        # fixed q value - two networks
        self.learning_rate = 0.0001
        self.fixed_q_value_steps = 100
        self.target_network_counter = 0

        self.PER_e = 0.01  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
        self.PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a
        self.PER_b = 0.4
        self.PER_b_increment = 0.005

        # n-step learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        if self.load_model:
            self.model = tf.keras.models.load_model('cartpole_nstep.h5')
            self.target_model = tf.keras.models.load_model('cartpole_nstep.h5')
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

    # DDDQN dueling double DQN, the network structure should change
    def create_model(self):
        inputs = tf.keras.Input(shape=(state_size,))
        fc1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
        advantage_output = tf.keras.layers.Dense(action_size, activation='linear')(fc2)
        if self.dueling:
            value_out = tf.keras.layers.Dense(1, activation='linear')(fc2)
            norm_advantage_output = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantage_output)
            outputs = tf.keras.layers.Add()([value_out, norm_advantage_output])
            model = tf.keras.Model(inputs, outputs)
        else:
            model = tf.keras.Model(inputs, advantage_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        return model

    def train(self, buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done, weights):
        if self.target_network_counter % self.fixed_q_value_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        buffer_action = np.squeeze(buffer_action).astype(np.int)
        buffer_reward = np.squeeze(buffer_reward)
        buffer_done = np.squeeze(buffer_done)
        # n_step learning: gamma is also truncated
        n_gamma = self.gamma ** self.n_step

        current_q = self.local_inference(buffer_state).numpy()
        # DDQN double DQN: choose action first in current network,
        # no axis=1 will only have one value
        max_next_q_indexes = np.argmax(self.local_inference(buffer_next_state).numpy(), axis=1)
        target_q = self.target_inference(buffer_next_state).numpy()
        target_q = target_q[np.arange(self.batch_size), max_next_q_indexes]
        # now the experience actually store n-step info
        # such as state[0], action[0], n-step reward, next_state[2] and done[2]
        target_q = np.where(buffer_done, buffer_reward,
                            buffer_reward + n_gamma * target_q).astype(np.float32)
        absolute_errors = current_q[np.arange(self.batch_size), buffer_action] - target_q
        current_q[np.arange(self.batch_size), buffer_action] = target_q
        self.train_batch(buffer_state, current_q, sample_weight=weights)
        return absolute_errors

    def compute_td_error(self, buffer_state, buffer_action, buffer_reward, buffer_next_state, buffer_done):
        buffer_action = np.squeeze(buffer_action).astype(np.int)
        buffer_reward = np.squeeze(buffer_reward)
        buffer_done = np.squeeze(buffer_done)
        batch_size = buffer_state.shape[0]

        n_gamma = self.gamma ** self.n_step

        current_q = self.local_inference(buffer_state).numpy()
        # DDQN double DQN: choose action first in current network,
        # no axis=1 will only have one value
        max_next_q_indexes = np.argmax(self.local_inference(buffer_next_state).numpy(), axis=1)
        target_q = self.target_inference(buffer_next_state).numpy()
        target_q = target_q[np.arange(batch_size), max_next_q_indexes]
        # now the experience actually store n-step info
        # such as state[0], action[0], n-step reward, next_state[2] and done[2]
        target_q = np.where(buffer_done, buffer_reward,
                            buffer_reward + n_gamma * target_q).astype(np.float32)
        absolute_errors = current_q[np.arange(batch_size), buffer_action] - target_q
        return absolute_errors

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

    def acting(self, state, test=False):
        if not test:
            # self.soft_update(self.model, self.target_model, 0.001)
            random_number = np.random.sample()
            if random_number > self.epsilon:
                action = np.argmax(self.local_inference(state[np.newaxis, ...]).numpy()[0])
            else:
                action = np.random.randint(action_size)

            return action
        return np.argmax(self.local_inference(state[np.newaxis, ...]).numpy()[0])


def get_weights_fn(policy):
    return (policy.model.get_weights(),
            policy.target_model.get_weights())


def set_weights_fn(policy, weights):
    model_weights, target_model_weights = weights
    policy.model.set_weights(model_weights)
    policy.target_model.set_weights(target_model_weights)


def explorer(global_rb, queue, trained_steps, is_training_done,
             lock, buffer_size=1024, episode_max_steps=1000):
    env = gym.make('CartPole-v1')
    policy = Agent()
    env_dict = {"obs": {"shape": (state_size,)},
                "act": {},
                "rew": {},
                "next_obs": {"shape": (state_size,)},
                "done": {}}
    local_rb = ReplayBuffer(buffer_size, env_dict=env_dict)
    local_idx = np.arange(buffer_size).astype(np.int)

    s = env.reset()
    episode_steps = 0
    total_reward = 0.
    total_rewards = []

    start = time.time()
    n_sample, n_sample_old = 0, 0

    while not is_training_done.is_set():
        n_sample += 1
        episode_steps += 1
        a = policy.acting(s)
        s_, r, done, _ = env.step(a)
        done_flag = done
        if episode_steps == env._max_episode_steps:
            done_flag = False
        total_reward += r
        policy.n_step_buffer.append((s, a, r, s_, done_flag))
        if len(policy.n_step_buffer) == policy.n_step:
            reward, next_state, done = policy.get_n_step_info(policy.n_step_buffer, policy.gamma)
            state, action = policy.n_step_buffer[0][:2]
            local_rb.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)

        s = s_
        if done or episode_steps == episode_max_steps:
            s = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0
            episode_steps = 0

        if not queue.empty():
            set_weights_fn(policy, queue.get())

        if local_rb.get_stored_size() == buffer_size:
            samples = local_rb._encode_sample(local_idx)
            td_errors = policy.compute_td_error(
                samples["obs"], samples["act"], samples["rew"],
                samples["next_obs"], samples["done"])
            priorities = np.abs(np.squeeze(td_errors)) + 1e-6

            lock.acquire()
            global_rb.add(
                obs=samples["obs"], act=samples["act"], rew=samples["rew"],
                next_obs=samples["next_obs"], done=samples["done"],
                priorities=priorities)
            lock.release()
            local_rb.clear()

            ave_rew = (0 if len(total_rewards) == 0 else
                       sum(total_rewards) / len(total_rewards))

            total_rewards = []
            start = time.time()
            n_sample_old = n_sample


def learner(global_rb, trained_steps, is_training_done,
            lock, n_training, update_freq, evaluation_freq, queues):
    policy = Agent()
    # Wait until explorers collect transitions
    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="learner")
    writer = tf.summary.create_file_writer(output_dir)
    writer.set_as_default()

    while not is_training_done.is_set() and global_rb.get_stored_size() < policy.n_warmup:
        continue

    start_time = time.time()
    while not is_training_done.is_set():
        trained_steps.value += 1
        tf.summary.experimental.set_step(trained_steps.value)
        lock.acquire()
        samples = global_rb.sample(policy.batch_size, policy.PER_b)
        lock.release()
        td_errors = policy.train(
            samples["obs"], samples["act"], samples["rew"],
            samples["next_obs"], samples["done"], samples["weights"])

        lock.acquire()
        global_rb.update_priorities(
            samples["indexes"], np.abs(td_errors) + policy.PER_e)
        lock.release()

        # Put updated weights to queue
        if trained_steps.value % update_freq == 0:
            weights = get_weights_fn(policy)
            for i in range(len(queues) - 1):
                queues[i].put(weights)
            fps = update_freq / (time.time() - start_time)
            tf.summary.scalar(name="apex/fps", data=fps)
            start_time = time.time()

        # Periodically do evaluation
        if trained_steps.value % evaluation_freq == 0:
            queues[-1].put(get_weights_fn(policy))
            queues[-1].put(trained_steps.value)

        if trained_steps.value >= n_training:
            is_training_done.set()


def evaluator(is_training_done, queue,
              save_model_interval=int(1e6), n_evaluation=10, episode_max_steps=1000,
              show_test_progress=False):
    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="evaluator")
    writer = tf.summary.create_file_writer(
        output_dir, filename_suffix="_evaluation")
    writer.set_as_default()

    policy = Agent()
    model_save_threshold = save_model_interval
    # checkpoint = tf.train.Checkpoint(policy=policy)
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)

    while not is_training_done.is_set():
        n_evaluated_episode = 0
        # Wait until a new weights comes
        if queue.empty():
            continue
        else:
            set_weights_fn(policy, queue.get())
            trained_steps = queue.get()
            tf.summary.experimental.set_step(trained_steps)
            avg_test_return = 0.
            for _ in range(n_evaluation):
                n_evaluated_episode += 1
                episode_return = 0.
                obs = env.reset()
                done = False
                for _ in range(episode_max_steps):
                    action = policy.acting(obs, test=True)
                    next_obs, reward, done, _ = env.step(action)
                    if show_test_progress:
                        env.render()
                    episode_return += reward
                    obs = next_obs
                    if done:
                        break
                avg_test_return += episode_return
                # Break if a new weights comes
                if not queue.empty():
                    break
            avg_test_return /= n_evaluated_episode
            tf.summary.scalar(
                name="apex/average_test_return", data=avg_test_return)

            if trained_steps > model_save_threshold:
                model_save_threshold += save_model_interval
                # checkpoint_manager.save()
    # checkpoint_manager.save()


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S.%f', suffix=""):
    if suffix is not "":
        suffix = "_" + suffix
    time_str = datetime.datetime.now().strftime(time_format) + suffix
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        raise RuntimeError('directory not specified')
    return outdir


if __name__ == '__main__':
    SyncManager.register('PrioritizedReplayBuffer',
                         PrioritizedReplayBuffer)
    manager = SyncManager()
    manager.start()

    PER_e = 0.01  # epsilon -> pi = |delta| + epsilon transitions which have zero error also have chance to be selected
    PER_a = 0.6  # P(i) = p(i) ** a / total_priority ** a

    env_dict = {"obs": {"shape": (state_size,)},
                "act": {},
                "rew": {},
                "next_obs": {"shape": (state_size,)},
                "done": {}}
    global_rb = manager.PrioritizedReplayBuffer(memory_size, env_dict=env_dict, alpha=PER_a, eps=PER_e)

    n_explorer = multiprocessing.cpu_count() - 1

    n_queue = n_explorer
    n_queue += 1  # for evaluation
    queues = [manager.Queue() for _ in range(n_queue)]

    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    is_training_done = Event()

    # Lock
    lock = manager.Lock()

    # Shared memory objects to count number of samples and applied gradients
    trained_steps = Value('i', 0)

    tasks = []
    local_buffer_size = 100
    episode_max_steps = 200

    for i in range(n_explorer):
        tasks.append(Process(
            target=explorer,
            args=[global_rb, queues[i], trained_steps, is_training_done,
                  lock, local_buffer_size, episode_max_steps]))

    n_training = 20000
    param_update_freq = 100
    test_freq = 1000
    tasks.append(Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done,
              lock, n_training, param_update_freq,
              test_freq, queues]))

    # Add evaluator
    save_model_interval = 1000
    n_evaluation = 5
    tasks.append(Process(
        target=evaluator,
        args=[is_training_done, queues[-1], save_model_interval, n_evaluation, episode_max_steps]))

    for task in tasks:
        task.start()
    for task in tasks:
        task.join()
