import numpy as np

"""
 
 [-7 -6 -5 -4 -3] [-2 -1 0 | 1 2] [3  4  5  6  7]
  0   1  2  3  4    5  6 7 | 8 9   10 11 12 13 14 ... ... 57 | 58 59 60 61 62 63 64 65
 x & y = 2 * 8 + 10 / 0.2 = 66
   z   = 2 * 8 + 4 / 0.2 = 36


"""


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


class Util:
    def __init__(self, r, g, b, rc, gc, bc, map3d):
        self.r = r
        self.g = g
        self.b = b
        self.rc = rc
        self.gc = gc
        self.bc = bc
        self.map3d = map3d

    def add_floor(self):
        self.map3d[np.where(self.bc < 0)] = 1

    def add_ceil(self, bound):
        self.map3d[np.where(self.bc > bound)] = 1

    def add_wall(self, bound):
        self.map3d[np.where(self.rc < -bound)] = 1
        self.map3d[np.where(self.rc > bound)] = 1
        self.map3d[np.where(self.gc < -bound)] = 1
        self.map3d[np.where(self.gc > bound)] = 1

    def add_cylinder(self, center, radius, height):
        self.map3d[np.where(((self.rc-center[0])**2+(self.gc-center[1])**2 <= radius**2)*(self.bc <= height))] = 1

    def add_ball(self, center, radius):
        self.map3d[np.where((self.rc - center[0])**2+(self.gc-center[1])**2+(self.bc-center[2])**2 <= radius ** 2)] = 1

    def add_box(self, center, x, y, z):
        map_x = np.abs(self.rc - center[0]) <= x
        map_y = np.abs(self.gc - center[1]) <= y
        map_z = np.abs(self.bc - center[2]) <= z
        self.map3d[np.where(map_x * map_y * map_z)] = 1

    def add_fence_1(self, center1, center2, x, y, z, fence_pos_z, fence_width):
        self.add_box(center1, x, y, z)
        self.add_box(center2, x, y, z)
        fence_center = (np.array(center1) + np.array(center2)) / 2
        fence_center[2] = fence_pos_z
        fence_y = np.abs(center1[1] - center2[1]) / 2
        self.add_box(fence_center, x, fence_y, fence_width)

    def add_fence_2(self, center1, center2, x, y, z, fence_pos_z, fence_width):
        self.add_box(center1, x, y, z)
        self.add_box(center2, x, y, z)
        fence_center = (np.array(center1) + np.array(center2)) / 2
        fence_center[2] = fence_pos_z
        fence_x = np.abs(center1[0] - center2[0]) / 2
        self.add_box(fence_center, fence_x, y, fence_width)