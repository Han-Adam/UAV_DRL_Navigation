import numpy as np
import json
import os

path = os.path.dirname(os.path.realpath(__file__))
data = []
for i in range(10):
    with open(path+"/Record"+str(i)+'/rate.json', 'r') as f:
        # x = np.array(json.load(f))
        # print(x.shape)
        data.append(json.load(f))

data = np.array(data)
print(data.shape)
np.save(path+"/3_data.npy", data)
