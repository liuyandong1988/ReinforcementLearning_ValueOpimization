#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/4 13:51
import gym
from Agent import QAgent


def train_agent():

    # 1. 创建环境
    env = gym.make("MountainCar-v0")
    env.reset()
    env.render()
    # the certain random
    agent = QAgent(env,
                   memory_capacity=2000,    # experience memory
                   hidden_dim=256)
    print("Learning...")
    agent.learning(max_episodes=300,
                   batch_size=64,
                   gamma=0.95,
                   min_epsilon=0.01)

if __name__ == "__main__":
    train_agent()