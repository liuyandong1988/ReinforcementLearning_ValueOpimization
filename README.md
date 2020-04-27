# 强化学习: 优化价值函数

## 1. SARSA
SARSA的想法是，用**同一个策略**下产生的动作A的Q值替代V(St+1)。On-policy的方法。

## 2. DQN
DQN的想法是，产生目标的策略始终**选取最大的action**是off-policy的方法。
### 2.1 DQN和SARSA
（1）二者使用Q-value代替V-value更新网络；（2）sarsa产生样本数据的策略和目标策略是一样的， 是on-policy方法；（3）DQN产生样本数据的方法和目标策略的方法不一样，是off-policy的方法。
### 2.2 DQN的技巧：episode-greedy
使用episode-greedy的方法产生行为，探索和利用。
### 2.3 DQN的技巧：Experience replay
将agent与环境互动的经验值，state action reward next-state done 储存起来，批处理时随机取用。（1）提高了数据的利用率；（2）打破数据的相关性，提升了网络的训练速度.
### 2.4 DQN的技巧： Fixed Target-q
如果每次更新时，目标的model总是变化，即对相同的state输出不同的Q-value，造成网络的震荡。使用Fixed target-Q的方法，Target-model会延时更新网络参数，以抑制网络的震荡。
### 2.5 DQN的技巧： Double DQN
DQN的目标策略是选取最大行为的Q-value，而真实情况下，agent对行为的选取episode-greedy。所以DQN估计出的Qvalue会偏大。使用Double-DQN的方法，两个Q网络的参数有差别，所以对于同一个动作的评估也会有少许不同。我们选取评估出来较小的值来计算目标。
实践中doubleDQN需要两个网络，一个负责提供action，一个提供该action的Q值。Q_network提供action，target_Q_network提供这个action的Q值。以减小argmaxQ估计带来的偏差。**Double DQN和Fixed Target-Q都是双Model，联合使用**
### 2.6 DQN变形 Dueling DQN
Q-value = A + V
如果一个state的V-value不好，如何选择action，都不会有好的结果。选取有价值的state，比每个action在state下的执行效果更有意义。
