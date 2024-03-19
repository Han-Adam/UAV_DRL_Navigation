from Env.env import Env
from Agent.double_dqn import DoubleDQN
import os
import json
import argparse
import time


def main(index):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path+'/Record'+str(index)
    if not os.path.exists(path):
        os.makedirs(path)

    agent = DoubleDQN(path, s_dim=56)
    env = Env()
    total_steps = 0
    max_step = 500000
    episode = 0
    store_index = 0

    while agent.train_it < max_step:
        episode += 1
        s = env.reset()
        ep_r = 0
        init_error = env.target_pos - env.current_pos
        for ep_step in range(400):
            a = agent.get_action(s)
            s_, r, done = env.step(a)
            agent.store_transition(s, a, s_, r, done)
            s = s_
            total_steps += 1
            ep_r += r

            if agent.train_it % 1000 == 1:
                # agent.store_net(str(store_index))
                store_index += 1
            if done:
                break
        last_error = env.target_pos - env.current_pos
        if r > 0:
            print()
            print(ep_step, ' episode: ', episode,
                  ' train_step: ', agent.train_it,
                  ' init_error: ', init_error,
                  ' last_error: ', last_error,
                  ' reward: ', ep_r,
                  ' epsilon: ', agent.epsilon)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='the parser')
    # parser.add_argument('method', type=str)
    # args = parser.parse_args()
    for i in range(10):
        main(i)
