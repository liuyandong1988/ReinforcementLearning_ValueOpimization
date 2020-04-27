#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/4 11:41

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def dqn_model(input_dim, hidden_dim, out_dim):
    """
    Building a Neural Network Model Keras
    """
    inputs = Input(shape=(input_dim,), name='state')
    x = Dense(hidden_dim, activation='relu')(inputs)
    x = Dense(out_dim, activation='linear', name='action')(x)
    q_model = Model(inputs, x)
    q_model.summary()
    return q_model

