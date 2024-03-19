from env.env import NavigationEnv
import time
import numpy as np
import matplotlib.pyplot as plt

env = NavigationEnv(render=True)

# env.reset(base_pos=[-4.5, -4.5, 1], target_pos=[4.5, 4.5, 1.5])
env.reset(base_pos=[-4.5, -4.5, 1], target_pos=[4.5, 4.5, 1.5])
i = 0
while True:
    i += 1
    print(i, env.current_pos)
    env.step()
