# tensorflow2_cartpole_PER
An tensorflow2 implementation of Prioritized Experience Replay, Dueling DQN, and double DQN.

Solved in episode 140, in gym environment "Cartpole-v1" with 200 limit.

## The training scores are:

Episode 0       Average Score: 23.00    epsilon:0.99    beta: 0.40 

Running for 0.00 seconds

Episode 100     Average Score: 127.92   epsilon:0.01    beta: 0.58 

Running for 238.84 seconds

Episode 140     Average Score: 195.92   epsilon:0.01    beta: 0.66

problem solved in 140 episode in 399.01 seconds

## important hyperparameters:

self.linear_annealed # the decreased epsilon per step, impacting the converge speed. This value seems great.

self.PER_b_increment # the increment of beta in prioritized experience replay. This value should be close to 1 with the training process.

self.fixed_q_value_steps # update period of target network weights. 

## dependencies

#### tensorflow==2.1.0
#### gym
#### numpy
#### matplotlib

## The result
