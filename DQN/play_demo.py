import gym
import q_learning as ql
import numpy as np
import time

def main():
    # env
    env = gym.make('CartPole-v0')
    # model
    input_dim = env.observation_space.shape[0]
    hidden_dim = 256
    output_dim = env.action_space.n
    model = ql.dqn_model(input_dim, hidden_dim, output_dim)
    # load the weight
    weight_file = './model/model_mountainCar.h5'
    model.load_weights(weight_file)
    # show the demo
    state = env.reset()
    s0 = np.reshape(state, [1, input_dim])
    total_reward = 0
    is_done = False
    while not is_done:
        a0 = int(np.argmax(model.predict(s0)[0]))
        s1, r1, is_done, info = env.step(a0)
        env.render()
        time.sleep(0.02)
        s1 = np.reshape(s1, [1, input_dim])
        s0 = s1
        total_reward += r1
    print('The reward: %d' %total_reward)







if __name__ == "__main__":
    main()