from Env.env import Env
from Agent.double_dqn import DoubleDQN
import os
import json
import argparse
import time
import numpy as np


path = os.path.dirname(os.path.realpath(__file__))

agent = DoubleDQN(path)
agent.epsilon = 1
agent.load_net('/Record9/470')
env = Env()

pos_record = []
s = env.reset(initial=[11, 11, 11], target=[158, 158, 29])
# s = env.reset(initial=[11, 58, 29], target=[58, 15, 11])

length = 0
start = time.time()
for ep_step in range(400):
    norm = np.max(np.abs(s[0:3]))
    if norm > 5:
        s[0:3] = s[0:3] / norm * 5
    pos_record.append(env.current_pos.tolist())
    a = agent.get_action(s)
    s_, r, done = env.step(a)
    # length += np.linalg.norm(s[0:3]-s_[0:3])
    # print(ep_step, r, env.current_pos, a, env._pos_value(env.current_pos), length, s[0:3], s_[0:3])
    s = s_
    if np.abs(r) > 10:
        print(r)
        break
    pos_record.append(env.current_pos.tolist())
print(time.time() - start)
# exit()

# np.save(path+'/3_2_Single_Path.npy', np.array(pos_record))

print(length*0.2)
# env.show_map(np.array(pos_record))



