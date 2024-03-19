import numpy as np
from qpsolvers import solve_qp
# import matplotlib.pyplot as plt


class MinimumAcc:
    def __init__(self, time_step, way_points, v_initial, a_initial, v_interval, v_end, a_end, psi_initial):
        self.time_step = time_step
        self.num_points = len(way_points)
        self.trajectory_pieces = self.num_points - 1
        self.order = 5
        # scatter points
        self.way_points = np.array(way_points)
        self.x, self.y, self.z = self.get_points()
        # set velocity and time
        self.v_initial = v_initial
        self.a_initial = a_initial
        self.v_interval = v_interval
        self.v_end = v_end
        self.a_end = a_end
        self.t = self.get_time_stamps()
        self.psi_intial = psi_initial
        # optimization parameters
        self.Q = self.get_Q()
        self.A = self.get_A()
        self.b_x, self.b_y, self.b_z = self.get_b()
        self.q = np.zeros(((self.order + 1) * self.trajectory_pieces, 1))\
                 .reshape(((self.order + 1) * self.trajectory_pieces,))
        self.G = np.zeros((4 * self.trajectory_pieces + 2, (self.order + 1) * self.trajectory_pieces))
        self.h = np.zeros((4 * self.trajectory_pieces + 2, 1)).reshape((4 * self.trajectory_pieces + 2,))

    def get_points(self):
        x = []
        y = []
        z = []
        for point in self.way_points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        return x, y, z

    def get_time_stamps(self):
        if self.trajectory_pieces == 1:
            print("only one trajectory")
            distance = np.linalg.norm(self.way_points[0, :] - self.way_points[1, :])
            t = [distance/1]
        else:
            t = []
            for i in range(self.trajectory_pieces):
                distance = np.linalg.norm(self.way_points[i, :] - self.way_points[i+1, :])
                t.append(distance/self.v_interval)
        return np.array(t)

    def get_Q(self):
        k = self.trajectory_pieces  # k pieces of trajectories, k+1 points
        n = self.order              # n-order ploy function
        Q = np.zeros(shape=[k * (n + 1), k * (n + 1)])  # row, column : diagonal matrix of k of [n+1, n+1]
        t = self.t

        for l in range(self.trajectory_pieces):
            for i in range(self.order - 1):
                for j in range(self.order - 1):
                    Q[l * (n + 1) + i, l * (n + 1) + j] = (n - i) * (n - i - 1) * (n - j) * (n - j - 1) / (
                                2 * n - i - j - 3) * t[l] ** (2 * n - i - j - 3)

        # adding the term to ensure that Q is a positive definite matrix
        Q = Q + (0.001 * np.identity((self.trajectory_pieces * (n+1))))
        return Q

    def get_A(self):
        # p = [p_n, p_n-1, ..., p1, p0]
        k = self.trajectory_pieces  # k pieces of trajectories, k+1 points
        n = self.order  # n-order ploy function
        A = np.zeros(shape=(4 * k + 2, k * (n + 1)))  # row, column
        t = self.t

        A[0, n - 1] = 1
        A[1, n - 2] = 2
        for i in range(k - 1):
            A[i * 4 + 2][(n + 1) * i + n] = 1  # fi_0 = xi
            for j in range(n + 1):
                A[i * 4 + 3][(n + 1) * i + j] = t[i] ** (n - j)  # fi_ti = x(i+1)
            for j in range(n):
                A[i * 4 + 4][(n + 1) * i + j] = (n - j) * t[i] ** (n - 1 - j)  # dfi_t1 = df(i+1)_0
            A[i * 4 + 4][(n + 1) * i + 2 * n] = -1
            for j in range(n - 1):
                A[i * 4 + 5][(n + 1) * i + j] = (n - j) * (n - j - 1) * t[i] ** (n - 2 - j)  # dfi_t1 = df(i+1)_0
            A[i * 4 + 5][(n + 1) * i + 2 * n - 1] = -2

        A[k * 4 - 2][(n + 1) * (k - 1) + n] = 1
        for j in range(n + 1):
            A[k * 4 - 1][(n + 1) * (k - 1) + j] = t[k - 1] ** (n - j)
        for j in range(n):
            A[k * 4][(n + 1) * (k - 1) + j] = (n - j) * t[k - 1] ** (n - 1 - j)
        for j in range(n - 1):
            A[k * 4 + 1][(n + 1) * (k - 1) + j] = (n - j) * (n - j - 1) * t[k - 1] ** (n - 1 - j)
        return A

    def get_b(self):
        """
        n = 3, k=3 as example
                                  0      1      2      3      4   5   6   7   8   9   10  11
        0  | df0_0 = v_initial    0    + 0    + 0    + 0    + 1 + 0                           = v_initial
        1  | ddf0_0 = a_initial   0    + 0    + 0    + 2    + 0 + 0                           = a_initial
        -----------------
        2  | f0_0 = x0            0    + 0    + 0    + 0    + 0 + 1                           = x0          k=0
        3  | f0_t0 = x1           t5   + t4   + t3   + t2   + t + 1                           = x1
        4  | df0_t0 = df1_0       5t^4 + 4t^3 + 3t^2 + 2t^1 + 1 + 0 [- 0 - 0 - 0 - 0 - 1 - 0] = 0
        5  | ddf0_t0 = ddf1_0     54t3 + 43t2 + 32t1 + 2    + 0 + 0 [- 0 - 0 - 0 - 2 - 0 - 0] = 0
        -----------------
        6  | f1_0 = x1                                                                                      k=1
        7  | f1_t1 = x2
        8  | df1_t1 = df2_0
        9  | ddf1_t1 = ddf2_0
        -----------------
        10 | f2_0 = x2                                                                                       k=2
        11 | f2_t2 = x3
        12 | df2 = v_end
        13 | ddf2 = a_end
        """
        points = [self.x, self.y, self.z]
        b = []
        for i in range(3):
            bi = [self.v_initial[i], self.a_initial[i]]
            for j in range(self.trajectory_pieces):
                bi.append(points[i][j])
                bi.append(points[i][j + 1])
                bi.append(0)
                bi.append(0)
            bi[-2] = self.v_end[i]
            bi[-1] = self.a_end[i]
            b.append(bi)

        print(b)
        return b

    def solve(self):
        """
        min x^T Q x
        A x = b
        """
        self.p_x = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_x, solver='cvxpy')
        self.p_y = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_y, solver='cvxpy')
        self.p_z = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_z, solver='cvxpy')

    def get_trajectory(self):
        time_resolution = np.floor(self.t / self.time_step)
        time_resolution = np.array(time_resolution, dtype=np.compat.long)
        n = self.order
        self.solve()
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z
        x, y, z, psi = [], [] , [], []
        for i in range(self.trajectory_pieces):
            x_segment = []
            y_segment = []
            z_segment = []
            t = np.linspace(0, self.t[i], time_resolution[i])
            for j in range(time_resolution[i]):
                x_term, y_term, z_term = 0, 0, 0
                for l in range(n + 1):
                    x_term = x_term + p_x[(n + 1) * i + l] * (t[j] ** (n - l))
                    y_term = y_term + p_y[(n + 1) * i + l] * (t[j] ** (n - l))
                    z_term = z_term + p_z[(n + 1) * i + l] * (t[j] ** (n - l))
                x_segment.append(x_term)
                y_segment.append(y_term)
                z_segment.append(z_term)
            x = np.concatenate([x, x_segment])
            y = np.concatenate([y, y_segment])
            z = np.concatenate([z, z_segment])
            psi = np.concatenate([psi, np.linspace(self.psi_intial if i==0 else psi[-1],
                                                   np.arctan2(self.y[i+1]-self.y[i], self.x[i+1]-self.x[i]),
                                                   time_resolution[i])])

        print(psi.shape, x.shape)
        trajectory = np.vstack([x, y, z, psi]).T

        return trajectory