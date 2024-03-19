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
s = env.reset(initial=[11, 11, 11], target=[58, 58, 29])
# s = env.reset(initial=[11, 58, 29], target=[58, 15, 11])

length = 0
r_record = -20
start_t = time.time()
for ep_step in range(400):
    # pos_record.append(env.current_pos.tolist())
    a = agent.get_action(s)
    s_, r, done = env.step(a)
    length += np.linalg.norm(s[0:3]-s_[0:3])
    r_record += r
    s = s_
    # print(ep_step, r, env.current_pos, a, env._pos_value(env.current_pos))
    if np.abs(r) > 10:
        break
    pos_record.append(env.current_pos.tolist())
print(time.time() - start_t)

exit()

# np.save(path+'/3_2_Single_Path.npy', np.array(pos_record))

print(length)
print(r_record*0.2)
print(np.linalg.norm([11, 11, 11] - np.array([58, 58, 29])) * 0.2)

length = 0
x = np.array(pos_record)
for i in range(x.shape[0]-1):
    print(x[i, :])
    length += np.linalg.norm(x[i, :] - x[i+1, :])
print(length*0.2)


env.show_map(np.array(pos_record))



