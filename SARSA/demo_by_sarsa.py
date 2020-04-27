from main_env import maze_grid
import sarsa
import time



def get_Q_table():
    # read from local
    f = open("sarsa_q.txt", 'r')
    q_table = eval(f.read())
    f.close()
    return q_table



def demo_main():
    # 1. 环境
    env = maze_grid('sarsa')
    env.render()
    # 2. agent by SARSA 下载Q-table表，策略
    agent = sarsa.Agent(env)
    agent.Q = get_Q_table()
    print("Test...")
    # 3. agent 与环境交互
    max_times = 3
    for i in  range(max_times):
        print('Test time: %d / %d' %(i+1, max_times))
        # 3.1 初始化环境信息
        state = env.reset()  # 环境初始化
        s0 = agent._get_state_name(state)  # 获取个体对于观测的命名
        while True:
            # 3.2 局部观测得到action
            Q_s = agent.Q[s0]
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
            # 3.3 与环境交互
            s1, r1, is_done, info = agent.act(action)
            env.render()
            time.sleep(0.1)
            s1 = agent._get_state_name(s1)
            # 3.4 状态转移
            s0 = s1
            # 3.5 判断是否完成任务
            if is_done:
                print('Episode %d finishes !'%i)
                break
    print('Game over !')
    input('123')





if __name__ == '__main__':
    demo_main()