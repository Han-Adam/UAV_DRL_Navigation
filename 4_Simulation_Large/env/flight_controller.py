import numpy as np


class SE3:
    def __init__(self, time_step):
        self.K_pos = np.array([2.9, 2.9, 4])
        self.K_vel = np.array([1.6, 1.6, 2])
        self.K_R = np.array([30, 30, 30])
        self.K_omega = np.array([3.5, 3.5, 3.5])
        self.G = 9.81
        self.M = 0.027
        self.J = [[1.40e-5, 0, 0],
                  [0, 1.40e-5, 0],
                  [0, 0, 2.17e-5]]
        self.time_step = time_step
        self.t = None
        self.pos_d = self.vel_d = self.acc_d = None

    def reset(self, trajectory):
        trajectory = trajectory.tolist()
        for i in range(2):
            trajectory.append(trajectory[-1])
        self.pos_d = np.array(trajectory)
        self.vel_d = (self.pos_d[1:, :] - self.pos_d[:-1, :]) / self.time_step
        self.acc_d = (self.vel_d[1:, :] - self.vel_d[:-1, :]) / self.time_step
        self.t = 0

    def computControl(self, pos, vel, matrix, ang_vel):
        b3 = matrix[:, 2]
        pos_d = self.pos_d[self.t, :3]
        vel_d = self.vel_d[self.t, :3]
        acc_d = self.acc_d[self.t, :3]

        psi_d = self.pos_d[self.t, 3]
        psi_d_d = np.array([0, 0, self.vel_d[self.t, 3]])

        pos_a = np.clip(self.K_pos * (pos_d - pos) + self.K_vel * (vel_d - vel) + acc_d/4, -1, 1)
        f_d = self.M * ([0, 0, self.G] + 4 * pos_a)
        thrust = f_d[2]/matrix[2, 2]
        # thrust = np.dot(b3, f_d)

        f_d_norm = np.linalg.norm(f_d)
        b3_d = np.array([0, 0, 1]) if f_d_norm <= 0 else f_d / f_d_norm
        b_psi = np.array([np.cos(psi_d), np.sin(psi_d), 0])
        b2_d = np.cross(b3_d, b_psi)
        b2_d /= np.linalg.norm(b2_d)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.array([b1_d, b2_d, b3_d]).T

        e_R = (np.matmul(R_d.T, matrix) - np.matmul(matrix.T, R_d)) / 2
        e_R = np.array([e_R[1, 2], e_R[2, 0], e_R[0, 1]])
        tau = 30 * np.matmul(self.J, np.clip(self.K_R * e_R + self.K_omega * (psi_d_d - ang_vel), -1, 1))
        self.t += 1
        return np.array([thrust, tau[0], tau[1], tau[2]]), pos_d
