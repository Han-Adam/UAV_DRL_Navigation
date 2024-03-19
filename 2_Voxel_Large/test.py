import numpy as np

from Env.env import Env
from Agent.double_dqn import DoubleDQN
import os
import json
import argparse
import time

env = Env()
env.show_map(path=[])
exit()

def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path = path+'/Record'+str(9)

    agent = DoubleDQN(path, s_dim=56)
    agent.epsilon = 1
    agent.load_net(prefix='470')
    env = Env()

    success_rate = []
    crash_rate = []
    trap_rate = []
    path_length = []
    for i in range(1, 31):
        success_count = 0
        crash_count = 0
        trap_count = 0
        path_length_count = []
        for j in range(2700):
            s = env.reset()
            path_length_record = 0
            for ep_step in range(400):
                s[0:3] = np.clip(s[0:3], -i, i)

                a = agent.get_action(s)
                s_, r, done = env.step(a)
                path_length_record += np.linalg.norm(env.current_pos-env.last_pos)
                s = s_

                if r > 10:
                    success_count += 1
                    path_length_count.append(path_length_record)
                    break
                elif r < -10:
                    crash_count += 1
                    break

            if np.abs(r) < 10:
                trap_count += 1
        path_length_count = np.mean(path_length_count) if len(path_length_count)>0 else 0
        print(0, i, success_count, crash_count, trap_count, path_length_count)
        success_rate.append(success_count)
        crash_rate.append(crash_count)
        trap_rate.append(trap_count)
        path_length.append(path_length_count)

    rate = np.vstack([success_rate, crash_rate, trap_rate, path_length]).tolist()
    # with open(path + '/clip_rate.json', 'w') as f:
    #     json.dump(rate, f)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='the parser')
    # parser.add_argument('method', type=str)
    # args = parser.parse_args()
    main()
