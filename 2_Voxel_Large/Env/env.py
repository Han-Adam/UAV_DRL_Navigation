import copy

import numpy as np
from .util import Util, midpoints
import matplotlib.pyplot as plt
# from mayavi import mlab


class Env:
    def __init__(self):
        # r, g, b = x, y, z
        self.r, self.g, self.b = np.indices((71, 71, 41)) * 0.2
        self.r -= 7
        self.g -= 7
        self.b -= 2
        self.rc = midpoints(self.r)
        self.gc = midpoints(self.g)
        self.bc = midpoints(self.b)

        util = Util(self.r, self.g, self.b, self.rc, self.gc, self.bc, np.zeros(shape=[70, 70, 40]))
        util.add_floor()
        util.add_ceil(4)
        util.add_wall(5)

        util.add_fence_1([-4, -4, 2], [-4, -2, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_2([-4, -2, 2], [-2, -2, 2], 0.1, 0.1, 2, 3.5, 0.5)
        util.add_fence_1([-2, -4, 2], [-2, -2, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_1([-2, -4, 2], [-2, -2, 2], 0.1, 0.1, 2, 3, 0.5)
        util.add_cylinder([-3, -3], 0.4, 3)

        util.add_box([-3, -1, 3.5], 0.1, 1, 0.5)

        util.add_fence_1([-4, -1, 2], [-4, 1, 2], 0.1, 0.1, 2, 2, 0.5)
        util.add_fence_1([-2, -1, 2], [-2, 1, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_1([-2, -1, 2], [-2, 1, 2], 0.1, 0.1, 2, 3, 0.5)
        util.add_box([-3, 0, 2], 1, 0.1, 2)

        # util.add_box([-3.1, 1, 1.5], 0.1, 1, 0.5)

        for i in [-4, -3, -2]:
            for j in [4, 3, 2]:
                util.add_cylinder([i, j], 0.2, 4)

        util.add_fence_2([-1, 4, 2], [2, 4, 2], 0.1, 0.1, 2, 2, 0.5)
        util.add_fence_2([-1, 3, 2], [2, 3, 2], 0.1, 0.1, 2, 3.5, 0.5)
        util.add_fence_2([-1, 3, 2], [2, 3, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_2([-1, 1, 2], [2, 1, 2], 0.1, 0.1, 2, 3.5, 0.5)
        util.add_fence_2([-1, 1, 2], [2, 1, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_1([-1, 2, 2], [-1, 3, 2], 0.1, 0.1, 2, 2, 2)
        util.add_fence_1([0, 2, 2], [0, 1, 2], 0.1, 0.1, 2, 2, 2)
        util.add_fence_1([1, 2, 2], [1, 3, 2], 0.1, 0.1, 2, 2, 2)
        util.add_fence_1([2, 2, 2], [2, 1, 2], 0.1, 0.1, 2, 2, 2)

        util.add_fence_1([-1, 1, 2], [-1, 0, 2], 0.1, 0.1, 2, 2, 2)
        util.add_fence_1([2, 1, 2], [2, 0, 2], 0.1, 0.1, 2, 2, 2)

        # util.add_fence_2([-1, 0, 2], [2, 0, 2], 0.1, 0.1, 2, 2, 1)
        for i in [-0, 1]:
            for j in [-1, -2]:
                util.add_cylinder([i, j], 0.4, 3)
        util.add_fence_2([-2, -1, 1.5], [0, -1, 1.5], 0.1, 0.1, 1.5, 1.5, 0.5)
        util.add_fence_2([1, -1, 1.5], [3, -1, 1.5], 0.1, 0.1, 1.5, 1.5, 0.5)
        util.add_fence_1([0, -2, 1.5], [0, -4, 1.5], 0.1, 0.1, 1.5, 1.5, 0.5)
        util.add_fence_1([1, -2, 1.5], [1, -4, 1.5], 0.1, 0.1, 1.5, 0.5, 0.5)
        util.add_fence_2([-2, -4, 2], [0, -4, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_2([-2, -4, 2], [0, -4, 2], 0.1, 0.1, 2, 2.5, 0.5)
        util.add_cylinder([-1, -3], 0.4, 3)

        for i in [3, 4]:
            for j in [2, 3, 4]:
                util.add_cylinder([i, j], 0.2, 4)
        # util.add_fence_2([2, 2, 2], [3, 2, 2], 0.1, 0.1, 2, 1, 1)
        util.add_fence_1([3, 2, 2], [3, 3, 2], 0.1, 0.1, 2, 2, 1)
        util.add_fence_2([3, 3, 2], [4, 3, 2], 0.1, 0.1, 2, 1, 1)
        util.add_fence_1([4, 3, 2], [4, 4, 2], 0.1, 0.1, 2, 2, 1)

        util.add_fence_1([3, -1, 2], [3, 1, 2], 0.1, 0.1, 2, 3, 0.5)
        util.add_fence_1([3, -1, 2], [3, 1, 2], 0.1, 0.1, 2, 0.5, 0.5)
        util.add_fence_1([4, -1, 2], [4, 1, 2], 0.3, 0.3, 2, 2.5, 0.5)
        util.add_fence_2([3, 1, 2], [4, 1, 2], 0.1, 0.1, 2, 2, 2)
        # util.add_fence_2([3, -0.5, 2], [4, -0.5, 2], 0.1, 0.1, 2, 2.5, 1)

        util.add_fence_2([1, -2, 1.5], [4, -2, 1.5], 0.1, 0.1, 1.5, 0.5, 0.5)
        util.add_fence_2([1, -2, 1.5], [4, -2, 1.5], 0.1, 0.1, 1.5, 2.5, 0.5)
        # util.add_fence_1([4, -4, 2], [4, -2, 2], 0.1, 0.1, 2, 2, 1.3)
        # util.add_box([3, -3, 1], 1, 1, 0.1)
        # util.add_box([3, -3, 2], 1, 1, 0.1)
        # for i in range(-4, 5, 2):
        #     util.add_cylinder([-4, i], np.random.rand() * 0.3 + 0.5, np.random.rand() * 2 + 1)
        #     util.add_cylinder([4, i], np.random.rand() * 0.5 + 0.4, np.random.rand() * 2 + 1)
        # util.add_ball([0, 2, 1.5], 0.8)
        # util.add_ball([0, -2, 1.5], 0.8)
        # util.add_fence_1([-2, 4, 1.5], [-2, 2, 1.5], 0.6, 0.4, 1.5, 2.5, 0.5)
        # util.add_fence_2([0, 4, 1.5], [2, 4, 1.5], 0.4, 0.4, 1.5, 1.5, 0.5)
        # util.add_fence_2([-2, 0, 1.5], [0, 0, 1.5], 0.4, 0.4, 1.5, 2.5, 0.5)
        # util.add_fence_2([0, 0, 1.5], [2, 0, 1.5], 0.4, 0.4, 1.5, 0.5, 0.5)
        # util.add_fence_1([2, 2, 1.5], [2, -2, 1.5], 0.4, 0.6, 1.5, 2.5, 0.5)
        # util.add_fence_1([-2, -4, 1.5], [-2, -2, 1.5], 0.2, 0.6, 1.5, 0.5, 0.5)
        # util.add_fence_2([0, -4, 1.5], [2, -4, 1.5], 0.2, 0.6, 1.5, 2.5, 0.5)

        map3d_small = util.map3d

        self.r, self.g, self.b = np.indices((171, 171, 41)) * 0.2
        self.r -= 17
        self.g -= 17
        self.b -= 2
        self.rc = midpoints(self.r)
        self.gc = midpoints(self.g)
        self.bc = midpoints(self.b)
        util = Util(self.r, self.g, self.b, self.rc, self.gc, self.bc, np.zeros(shape=[170, 170, 40]))
        util.add_floor()
        util.add_ceil(4)
        util.add_wall(15)
        self.map3d = util.map3d
        for i in range(3):
            for j in range(3):
                self.map3d[10+i*50:60+i*50, 10+j*50:60+j*50, 10:30] = map3d_small[10:60, 10:60, 10:30]

        self.current_pos = self.last_pos = None
        self.target_pos = None
        self.initial_pos = None

    def reset(self, initial=None, target=None):
        # initial position
        if initial is None:
            while 1:
                x = np.random.randint(10, 160) # [low, high)
                y = np.random.randint(10, 160)
                z = np.random.randint(10, 30)
                if self.map3d[x, y, z] == 0:
                    self.initial_pos = self.current_pos = np.array([x,y,z])
                    break
        elif self._pos_value(initial) == 0:
            self.initial_pos = self.current_pos = np.array(initial)
        else:
            print("Initial position is occupied")
            exit()
        # target position
        if target is None:
            while 1:
                x = np.random.randint(10, 160)  # [low, high)
                y = np.random.randint(10, 160)
                z = np.random.randint(10, 30)
                if self.map3d[x, y, z] == 0 and np.any(self.current_pos != [x, y, z]):
                    self.target_pos = np.array([x, y, z])
                    break
        elif self._pos_value(target) == 0:
            self.target_pos = np.array(target)
        else:
            print("Target position is occupied")
            exit()

        return self._get_s()

    def step(self, a):
        self.last_pos = self.current_pos

        x, y, z = self.current_pos
        a += 1 if a >= 13 else 0
        x_ = x - int(a / 9) + 1
        y_ = y - int(a % 9 / 3) + 1
        z_ = z - a % 3 + 1
        self.current_pos = np.array([x_, y_, z_])

        s = self._get_s()
        r = self._get_r()
        done = np.abs(r) > 10
        return s, r, done

    def _get_s(self):
        s_error = self.target_pos - self.current_pos
        # norm = np.linalg.norm(s_error)
        # if norm > 20:
        #     s_error = np.round(s_error / norm * 20)
        s_error = (0.2 * s_error).tolist()

        x, y, z = self.current_pos
        s_accurate = self.map3d[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        s_accurate = np.reshape(s_accurate, [-1])

        s_statistic = []
        for i in range(27):
            if i == 13:
                continue
            x_ = x - 3 * int(i / 9) + 3
            y_ = y - 3 * int(i % 9 / 3) + 3
            z_ = z - 3 * (i % 3) + 3
            s_statistic.append(np.mean(self.map3d[x_ - 1:x_ + 2, y_ - 1:y_ + 2, z_ - 1:z_ + 2]))

        return np.concatenate([s_error, s_accurate, s_statistic])

    def _get_r(self):
        r1 = (np.linalg.norm(self.last_pos-self.target_pos) - np.linalg.norm(self.current_pos-self.target_pos)) # * 0.2
        r2 = 20 if np.all(self.current_pos == self.target_pos) else 0
        r3 = -20 if self._pos_value(self.current_pos) == 1 else 0
        # r4 = 0 # -np.abs(self.current_pos[2] - self.target_pos[2])
        return r1 + r2 + r3 # + r4

    def _pos_value(self, index):
        return self.map3d[index[0], index[1], index[2]]

    def show_map(self, path):
        print('show')
        r, g, b = np.indices((151, 151, 151)) * 0.2
        r -= 15
        g -= 15
        # b -= 1
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)

        env_map = np.zeros(shape=[150, 150, 150])
        env_map[:, :, 0:20] = self.map3d[10:160, 10:160, 10:30]
        env_map = env_map == 1

        # path_map = np.zeros(shape=[150, 150, 150])
        # # for i in range(path.shape[0]):
        # #     path_map[path[i, 0] - 10, path[i, 1] - 10, path[i, 2] - 10] = 1
        # path_map = path_map == 1

        # initial_map = np.zeros(shape=[150, 150, 150])
        # initial_map[self.initial_pos[0] - 10, self.initial_pos[1] - 10, self.initial_pos[2] - 10] = 1
        # initial_map = initial_map == 1

        # target_map = np.zeros(shape=[150, 150, 150])
        # target_map[self.target_pos[0] - 10, self.target_pos[1] - 10, self.target_pos[2] - 10] = 1
        # target_map = target_map == 1

        colors = np.zeros(env_map.shape + (3,))
        colors[:, :, 0:6, 0] = np.linspace(2, 3, 6)
        colors[:, :, 0:6, 1] = np.linspace(32, 177, 6)
        colors[:, :, 0:6, 2] = np.linspace(97, 237, 6)
        colors[:, :, 6:20, 0] = np.linspace(3, 255, 14)
        colors[:, :, 6:20, 1] = np.linspace(177, 255, 14)
        colors[:, :, 6:20, 2] = np.linspace(237, 0, 14)
        colors /= 255

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(azim=-60, elev=40)
        # ax.view_init(azim=-40, elev=40)
        # ax.view_init(azim=-90, elev=90)
        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.grid(False)

        ax.voxels(r, g, b, env_map,
                  facecolors=colors,
                  edgecolors='#EEEEEE',
                  alpha=1,
                  linewidth=0.1)

        # ax.voxels(r, g, b, path_map,
        #           facecolors='red',
        #           edgecolors='#EEEEEE',
        #           linewidth=0.5)

        # initial_pos = 0.2 * self.initial_pos - np.array([16.9, 16.9, 1.9])
        # # # ax.plot_surface(x + initial_pos[0], y + initial_pos[1], z + initial_pos[2])
        # ax.scatter(initial_pos[0], initial_pos[1], initial_pos[2],
        #            s=20, c='red', marker='o', edgecolors='gray', linewidth=0.5)
        #
        # target_pos = 0.2 * self.target_pos - np.array([16.9, 16.9, 1.9])
        # ax.scatter(target_pos[0], target_pos[1], target_pos[2],
        #            s=30, facecolors='red', marker='*', edgecolors='gray', linewidth=0.3)

        # ax.voxels(r, g, b, initial_map,
        #           facecolors='blue',
        #           edgecolors='#EEEEEE',
        #           linewidth=0.5)
        #
        # ax.voxels(r, g, b, target_map,
        #           facecolors='black',
        #           edgecolors='#EEEEEE',
        #           linewidth=0.5)

        ax.set(xlabel='x', ylabel='y', zlabel='z')
        fig.savefig('C:/Users/asus/Desktop/Test1.jpeg', dpi=2000)
        plt.show()