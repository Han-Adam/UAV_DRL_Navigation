import os
from .uav import UAV
from .surrounding import Surrounding
from .flight_controller import SE3
from .trajectory_optimizer import MinimumAcc
from .path_planner import PathPlanner
from .camera import Camera
from .util import Util, midpoints
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt


class NavigationEnv:
    def __init__(self,
                 model='cf2x',
                 render=False,
                 random=True,
                 time_step=0.01):
        '''
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        '''
        self.render = render
        self.model = model
        self.random = random
        self.time_step = time_step
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.client = None
        self.time = None
        self.counter_image = None
        self.counter_decision = None
        self.surr = None
        self.current_pos = self.last_pos = None
        self.current_ori = self.last_ori = None
        self.current_matrix = self.last_matrix = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target_pos = None
        self.initial_pos = None
        self.uav = None
        self.map3d = None
        self.path_map = None
        self.trajectory = None
        self.trajectory_optimizer = None
        self.camera = None
        self.path_planner = None
        self.flight_controller = None
        self.approaching = None
        self.pos_record = None

        self.camera_pos = -10

    def close(self):
        p.disconnect(self.client)

    def reset(self, base_pos, target_pos):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        if self.render:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=10.9,
                                         cameraYaw=0,
                                         cameraPitch=-89.99,
                                         cameraTargetPosition=[self.camera_pos, 0, 0],
                                         physicsClientId=self.client)
            # p.resetDebugVisualizerCamera(cameraDistance=8,  # 12.9,
            #                              cameraYaw=45,  # 0,
            #                              cameraPitch=-45,  # -89.99,
            #                              cameraTargetPosition=[1, -1, 1],
            #                              physicsClientId=self.client)
            # p.resetDebugVisualizerCamera(cameraDistance=5,
            #                              cameraYaw=0,
            #                              cameraPitch=-89.99,
            #                              cameraTargetPosition=[3, 0.1, 0],
            #                              physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        self.time = 0.
        self.counter_image = 0
        self.counter_decision = 0
        # 构建空地图
        self.r, self.g, self.b = np.indices((191, 91, 61)) * 0.2
        self.r -= 19
        self.g -= 9
        self.b -= 4
        self.rc = midpoints(self.r)
        self.gc = midpoints(self.g)
        self.bc = midpoints(self.b)
        util = Util(self.r, self.g, self.b, self.rc, self.gc, self.bc, np.zeros(shape=[190, 90, 60]))
        util.add_floor()
        util.add_ceil(4)
        util.add_wall(bound_x=15, bound_y=5)
        self.map3d = util.map3d
        self.path_map = np.zeros(shape=[190, 90, 60])
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        # 初始化时便最好用float
        base_pos = np.array(base_pos)
        base_ori = np.array([0., 0., 0.])
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_matrix = self.last_matrix = np.array([[1., 0., 0.],
                                                           [0., 1., 0.],
                                                           [0., 0., 1.]])
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.target_pos = np.array(target_pos)
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        self.camera = Camera(self.client)
        self.path_planner = PathPlanner(self.r, self.g, self.b, self.target_pos)
        self.flight_controller = SE3(time_step=self.time_step)

        self.approaching = True
        self.pos_record = []

        # pos_record = np.load(self.path+'/Free_NoTranc.npy')
        # for i in range(pos_record.shape[0] - 1):
        #     p.addUserDebugLine(lineFromXYZ=pos_record[i, :],
        #                        lineToXYZ=pos_record[i+1, :],
        #                        lineColorRGB=[0, 1, 0],
        #                        lineWidth=5)

    def step(self):
        error = self.current_pos - self.target_pos
        dist = np.linalg.norm(error)
        if dist < 0.2:
            print("we win")
            self._show_the_trace()
        elif self.approaching:
            if dist < 2:
                self.approaching = False
                way_points = np.vstack([self.current_pos, self.target_pos])
                self.trajectory_optimizer = MinimumAcc(time_step=self.time_step,
                                                       way_points=way_points,
                                                       v_initial=np.array(self.current_vel),
                                                       a_initial=(np.array(self.current_vel) -
                                                                  np.array(self.last_vel)) / self.time_step,
                                                       v_interval=1,
                                                       v_end=[0, 0, 0],
                                                       a_end=[0, 0, 0],
                                                       psi_initial=self.current_ori[2])

                self.trajectory = self.trajectory_optimizer.get_trajectory()
                self.flight_controller.reset(self.trajectory)
            else:
                if self.counter_decision % 100 == 0:
                    print(self.counter_decision)
                    self._path_plan()
                    self.flight_controller.reset(self.trajectory)
                if self.counter_image % 4 == 0:
                    self.counter_image = 0
                    if self.camera.map_refresh(self.r, self.g, self.b,
                                               self.map3d, self.current_pos,
                                               self.current_matrix, self.path_map):
                        print()
                        # self._path_plan()
                        # self.flight_controller.reset(self.trajectory)

        F, pos_d = self.flight_controller.computControl(self.current_pos,
                                                        self.current_vel,
                                                        self.current_matrix,
                                                        self.current_ang_vel)

        self.uav.apply_action(F, self.time)
        self.pos_record.append(self.current_pos.tolist())

        self.last_pos = self.current_pos
        self.last_ori = self.current_ori
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel
        self.last_matrix = self.current_matrix
        p.stepSimulation()
        self._check_hit()
        self.time += self.time_step
        self.counter_image += 1
        self.counter_decision += 1

        current_pos, current_ori = p.getBasePositionAndOrientation(self.uav.id)
        current_matrix = np.reshape(p.getMatrixFromQuaternion(current_ori), [3, 3])
        current_ori = p.getEulerFromQuaternion(current_ori)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ori = np.array(current_ori)
        self.current_matrix = current_matrix
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.matmul(current_ang_vel, current_matrix)
        p.addUserDebugLine(lineFromXYZ=self.last_pos,
                           lineToXYZ=self.current_pos,
                           lineColorRGB=[1, 0, 0],
                           lineWidth=5)
        print(self.current_pos)

        p.resetDebugVisualizerCamera(cameraDistance=10.9,
                                     cameraYaw=0,
                                     cameraPitch=-89.99,
                                     cameraTargetPosition=[self.camera_pos, 0, 0],
                                     physicsClientId=self.client)
        self.camera_pos = np.clip(np.max([self.camera_pos, self.current_pos[0]]), -9, 9)

    def _path_plan(self):
        # np.save(self.path+'/Free_NoTranc.npy', self.pos_record)
        print('new plan')
        # self.counter_decision = 0
        self.path_map = np.zeros(shape=[190, 90, 60])
        self.path_planner.get_path(self.map3d, self.path_map, self.current_pos)
        path_index = np.argwhere(self.path_map == 1)
        way_points = 0.2 * path_index - np.array([18.9, 8.9, 3.9])
        way_points[0, :] = self.current_pos
        self.trajectory_optimizer = MinimumAcc(time_step=self.time_step,
                                               way_points=way_points,
                                               v_initial=np.array(self.current_vel),
                                               a_initial=(np.array(self.current_vel) - np.array(
                                                   self.last_vel)) / self.time_step,
                                               v_interval=0.6,
                                               v_end=[0, 0, 0],
                                               a_end=[0, 0, 0],
                                               psi_initial=self.current_ori[2])
        self.trajectory = self.trajectory_optimizer.get_trajectory()

    def _check_hit(self):
        x = p.getContactPoints(bodyA=self.uav.id)
        if x != ():
            print(f"hit happen!")
            self._show_the_trace()

    def _show_the_trace(self):
        # for i in range(len(self.pos_record) - 1):
        #     p.addUserDebugLine(lineFromXYZ=self.pos_record[i],
        #                        lineToXYZ=self.pos_record[i+1],
        #                        lineColorRGB=[1, 0, 0],
        #                        lineWidth=5)
        pos_record = np.array(self.pos_record)
        print('\n\n\n')
        print('tracking is over')
        print('total time step:', pos_record.shape[0] * 0.01)
        length = np.sum(np.sum((pos_record[0:-1, :] - pos_record[1:, :])**2, axis=1)**0.5)
        print('total trajectory length:', length)
        # print('direct length, 12.88m')

        r, g, b = np.indices((151, 151, 151)) * 0.2
        r -= 15
        g -= 5
        # b -= 1
        env_map = np.zeros(shape=[150, 150, 150])
        env_map[:, 0:50, 0:20] = self.map3d[10:160, 10:60, 10:30]
        env_map = env_map == 1
        env_map[:, :, 0] = env_map[:, :, 1]
        colors = np.zeros(env_map.shape + (3,))
        colors[:, :, 0:6, 0] = np.linspace(2, 3, 6)
        colors[:, :, 0:6, 1] = np.linspace(32, 177, 6)
        colors[:, :, 0:6, 2] = np.linspace(97, 237, 6)
        colors[:, :, 6:20, 0] = np.linspace(3, 255, 14)
        colors[:, :, 6:20, 1] = np.linspace(177, 255, 14)
        colors[:, :, 6:20, 2] = np.linspace(237, 0, 14)
        colors /= 255


        ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(azim=-90, elev=90)
        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.grid(False)
        ax.voxels(r, g, b, env_map,
                  facecolors=colors,
                  edgecolors=colors,
                  alpha=0.3,
                  linewidth=0.1)
        ax.set(xlabel='x', ylabel='y', zlabel='z')

        pos_record = np.array(self.pos_record)
        pos_record[:, 2] += 1
        ax.plot3D(pos_record[:, 0], pos_record[:, 1], pos_record[:, 2], 'r')

        plt.show()
        exit()



def _get_diff(ang, target):
    diff = (target - ang + np.pi) % (np.pi*2) - np.pi
    return diff