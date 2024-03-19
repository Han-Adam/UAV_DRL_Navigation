from Env.env import Env
from Agent.double_dqn import DoubleDQN
import os
import json
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path = path+'/Record'
    with open(path + '/success_rate_sat.json', 'r') as f:
        line1 = json.load(f)
    with open(path + '/success_rate_sat_4.json', 'r') as f:
        line2 = json.load(f)

    index = np.array(range(0, 1000, 5))
    plt.plot(index, line1, label='line1')
    plt.plot(index, line2, label='sat')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='the parser')
    # parser.add_argument('method', type=str)
    # args = parser.parse_args()
    main()
