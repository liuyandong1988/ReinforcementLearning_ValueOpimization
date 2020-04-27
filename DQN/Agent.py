#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/4 6:43

from gym import Env, spaces
import random
import q_learning as ql
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.optimizers import Adam
import time

class QAgent(object):

    def __init__(self, env: Env = None, memory_capacity=2000, hidden_dim=100):
        if env is None:
            raise Exception("agent should have an environment")
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.env = env
        # 经验回放buffer
        self.replay_counter = 0
        self.replay_buffer_capacity = memory_capacity
        self.replay_buffer = deque(maxlen=self.replay_buffer_capacity)

        # Q Network weights filename
        self.weights_file = 'dqn_mountainCar.h5'
        # the double DQN, q_model and q_target_model
        self.q_model = ql.dqn_model(self.input_dim, hidden_dim, self.output_dim)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = ql.dqn_model(self.input_dim, hidden_dim, self.output_dim)
        # copy Q Network params to target Q Network
        self._update_weights()

    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)

    # copy trained Q Network params to target Q Network
    def _update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    # compute Q_max
    def _get_target_q_value(self, next_state, reward, done):
        # matrix
        action_index = np.argmax(self.q_model.predict(next_state)[0])
        # target Q Network evaluates the action
        q_tmp_value = self.target_q_model.predict(next_state)
        q_value = q_tmp_value[[range(len(q_tmp_value))], action_index][0]
        # Q_max = reward + gamma * Q_ma
        q_target = reward + self.gamma * q_value * (~ done)
        return q_target

    def _learn_from_memory(self, batch_size):
        # Sample experience
        trans_pieces = random.sample(self.replay_buffer, batch_size)  # the transition <s0, a0, r1, is_done, s1>
        # 矩阵方式批处理 加快运算速度 [s0, a0, r1, s1, is_done]
        state = np.array([i[0][0] for i in trans_pieces]) # batch*4
        action = np.array([i[1] for i in trans_pieces]) # batch*1
        reward = np.array([i[2] for i in trans_pieces]) # batch*1
        next_state = np.array([i[3][0] for i in trans_pieces]) # batch*4
        done = np.array([i[4] for i in trans_pieces]) # batch*1
        # policy prediction for a given state
        q_values = self.q_model.predict(state)  # batch*2 action_dim
        # get Q_max
        q_target_value = self._get_target_q_value(next_state, reward, done)
        q_values[range(len(action)), action] = q_target_value
        # train the Q-network
        self.q_model.fit(state, q_values, verbose=0)
        loss = self.q_model.evaluate(state, q_values, verbose=0)
        # the target_net update
        if self.replay_counter % 100 == 0:
            self._update_weights()
        self.replay_counter += 1
        return loss

    def act(self, a0, s0):
        s1, r1, is_done, info = self.env.step(a0)
        s1 = s1.reshape(1,-1)
        # put the <s0, a0, r1, is_done, s1> in the memory
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        if s1[0][0] >= 0.4:
            r1 += 1
        self.replay_buffer.append([s0, a0, r1, s1, is_done])
        return s1, r1, is_done, info

    def learning(self, max_episodes=1000, batch_size=64, gamma=0.99, min_epsilon=0.1, weight_file=None):
        """
        epsilon-greed find the action and experience replay
        :return:
        """
        if weight_file:
            print("loading weights from file: %s" % (weight_file,))
            self.q_model.load_weights(weight_file)
            self.target_q_model.load_weights(weight_file)

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        self.gamma = gamma
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = min_epsilon
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(max_episodes))

        total_steps, step_in_episode, num_episode = 0, 0, 0
        steps_history, rewards_history = list(), list()

        while num_episode < max_episodes:
            # update exploration-exploitation probability
            self.update_epsilon()
            # update the epsilon
            step_in_episode, total_reward = 0, 0
            loss, mean_loss = 0, 0
            is_done = False
            self.state = self.env.reset()
            s0 = self.state.reshape(1,-1)
            start_time = time.time()
            while not is_done:
                a0 = self.perform_policy(s0, self.epsilon)
                # self.env.render()
                s1, r1, is_done, info = self.act(a0, s0)
                total_reward += r1
                step_in_episode += 1
                s0 = s1
                if is_done:
                    break
                # call experience relay
                if len(self.replay_buffer) > batch_size:
                    loss += self._learn_from_memory(batch_size)
            mean_loss = loss / step_in_episode
            print("episode: {:03d}/{:d} time_step:{:d} epsilon:{:3.2f}, loss:{:.5f}"
                  .format(num_episode+1, max_episodes, step_in_episode, self.epsilon, mean_loss))
            print('Episode reward: {:.2f} time: {:.2f}'.format(total_reward,time.time()-start_time))
            total_steps += step_in_episode
            num_episode += 1
            steps_history.append(total_steps)
            rewards_history.append(step_in_episode)

        print('Saving the model params...')
        file_name = './model/model_mountainCar.h5'
        self.q_model.save_weights(file_name)
        print('Finish training !')

        # plot training rewards
        plt.plot(steps_history, rewards_history)
        plt.xlabel('steps')
        plt.ylabel('running avg rewards')
        plt.show()

        # store the weight and score result
        self.save_weights()
        with open('step_history.txt', 'w') as f1:
            for i in steps_history:
                f1.write(str(i)+' ')
        with open('rewards_history.txt', 'w') as f2:
            for i in rewards_history:
                f2.write(str(i)+' ')
        print('Save the results...')

        return

    # decrease the exploration, increase exploitation
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def perform_policy(self, s, epsilon=None):
        """
        New action based on the Q_update net
        """
        Q_s = self.q_model.predict(s)[0]
        if epsilon is not None and random.random() < epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            return int(np.argmax(Q_s))

