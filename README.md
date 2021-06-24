##About
This project is some implementations of DQN based on Tensorflow2

###Learning environment

"CartPole-v1" in openAI gym

###Implemented Algorithms
- A2C (Advantage Actor Critic)
- Apex (Distributed Prioritized Experience Replay)
- c51 (Categorical DQN + Dueling DQN + Double DQN + Prioritized Experience Replay)
- nstep (N-Step DQN + Dueling DQN + Double DQN + Prioritized Experience Replay)

###Solved Requirements in cartpole
Considered solved when the average return is greater than or equal to 195.0 over 100 consecutive trials.

###Experiment results
For example, The training process in **nstep_dqn/nstep.py**

- Episode 100   Average Score: 146.78  	epsilon:0.01	per_beta: 0.40	step: 14678

- Episode 129	Average Score: 195.61	epsilon:0.01	per_beta: 0.40	step: 20471

Problem solved in episode 129 with 0 hours 0 minutes 31 seconds

![image](https://github.com/sorryformyself/tensorflow2_cartpole_PER/blob/master/nstep_dqn/cartpole_nstep.png)
