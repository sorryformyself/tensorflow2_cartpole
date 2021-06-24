约31秒满分（连续100回合平均分数>195)。使用了cpprb第三方库作为replay buffer，而不是用python语言中的deque写一个replay buffer
运行：python nstep.py

训练环境：来自gym box2d游戏CartPole，对应gym中CartPole-v1
算法：nstep dqn, prioritized experience replay, double dqn, dueling dqn
框架：tensorflow

nstep文件：算法逻辑

训练数据如下:
Episode 100     	Average Score: 146.78  	 epsilon:0.01	per_beta: 0.40	step: 14678
Running for 14678 steps with 0 hours 0 minutes 22 seconds
Episode 129	Average Score: 195.61	epsilon:0.01	per_beta: 0.40	step: 20471
problem solved in episode 129 with 0 hours 0 minutes 31 seconds

