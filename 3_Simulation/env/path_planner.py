import torch
import torch.nn as nn
import numpy as np
import os


class QNet(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(QNet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, a_num))

    def forward(self, s):
        return self.feature(s)


class PathPlanner:
    def __init__(self, r, g, b, target_pos):
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.Q = QNet(s_dim=56, hidden=128, a_num=26)
        self.Q.load_state_dict(torch.load(self.file_path + '/Q_Net.pth'))

        self.r = r
        self.g = g
        self.b = b
        self.target_pos = np.array(np.floor(target_pos * 5) + [25, 25, 0] + [10, 10, 10], dtype=np.compat.long)

    def get_path(self, map3d, pathmap, current_pos):
        current_pos = np.array(np.floor(current_pos * 5) + [25, 25, 0] + [10, 10, 10], dtype=np.compat.long)
        for i in range(5):
            s = self._get_s(map3d, current_pos)
            a = self._get_action(s)
            x, y, z = current_pos
            a += 1 if a >= 13 else 0
            x_ = x - int(a / 9) + 1
            y_ = y - int(a % 9 / 3) + 1
            z_ = z - a % 3 + 1
            current_pos = np.array([x_, y_, z_])
            pathmap[x_, y_, z_] = 1
        print(np.argwhere(pathmap==1))

    def _get_action(self, s):
        s = torch.tensor(s, dtype=torch.float)
        actions_value = self.Q(s)
        action = torch.argmax(actions_value)
        action = action.item()
        return action

    def _get_s(self, map3d, current_pos):
        s_error = self.target_pos - current_pos
        s_error = s_error * 0.2

        x, y, z = current_pos
        s_accurate = map3d[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        s_accurate = np.reshape(s_accurate, [-1])

        s_statistic = []
        for i in range(27):
            if i == 13:
                continue
            x_ = x - 3 * int(i / 9) + 3
            y_ = y - 3 * int(i % 9 / 3) + 3
            z_ = z - 3 * (i % 3) + 3
            s_statistic.append(np.mean(map3d[x_ - 1:x_ + 2, y_ - 1:y_ + 2, z_ - 1:z_ + 2]))

        return np.concatenate([s_error, s_accurate, s_statistic])
