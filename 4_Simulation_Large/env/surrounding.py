import pybullet as p
import pybullet_data
import numpy as np


class Surrounding(object):
    def __init__(self,
                 client=0,
                 time_step=0.01,
                 g=9.81):
        self.client = client
        self.time_step = time_step
        self.G = g
        self.construct()

    def construct(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane100.urdf", physicsClientId=self.client)
        # add_fence(center_pos=[0, 0, 2], internal_length=10.4, internal_width=10.4, height=4, thickness=1)
        self.surr_0()
        p.setGravity(0., 0., -self.G)
        p.setTimeStep(self.time_step)

    def surr_0(self):
        add_fence([0, 0, 2], 10.2, 30.2, 4, 1)

        for i in range(-14, -9):
            for j in range(-3, 5):
                add_box([i, j, 2], halfExtents=[0.1, 0.1, 2])
        add_box([-12, -1.5, 1.5], halfExtents=[2.5, 0.2, 1.5])

        for i in [-9, -7]:
            for j in [-3, -1, 1, 3]:
                add_box([i, j, 1.5], halfExtents=[0.46, 0.46, 1.5])
        add_box([-6, -0, 1], halfExtents=[0.2, 5, 1])

        add_box([-4.5, -0.1, 1], halfExtents=[0.2, 0.1, 1])
        add_box([-4.5, -4, 1], halfExtents=[0.2, 1, 1])
        add_box([-4.5, -2.45, 2.5], halfExtents=[0.2, 2.45, 0.5], )
        add_box([-4.3, 2.5, 2], halfExtents=[0.4, 2.5, 2])

        for j in [-3, -1, 1]:
            add_box([-3, j, 1.5], halfExtents=[0.5, 0.5, 1.5])
        add_box([-3, 3.75, 1.5], halfExtents=[0.5, 1.25, 1.5])
        for j in [-4, -2, 0, 2]:
            add_box([-1, j, 1.5], halfExtents=[0.5, 0.5, 1.5])
        add_box([-1, 3.75, 1.5], halfExtents=[1.5, 1.25, 1.5])
        add_box([0, 1.75, 1.5], halfExtents=[0.5, 0.75, 1.5])
        # for j in [-3, -1]:
        #     add_box([1, j, 1.5], halfExtents=[0.5, 0.5, 1.5])
        add_box([1, 2.9, 1.5], halfExtents=[0.5, 2.1, 1.5])
        add_box([3, 2.8, 1.5], halfExtents=[1.5, 2.2, 1.5])
        add_box([2.25, -2.4, 1.5], halfExtents=[2.25, 2.3, 1.5])

        add_box([5.9, 0, 1], halfExtents=[0.2, 5, 1])

        for j in [-3, -1, 1, 3]:
            add_box([7, j, 1.5], halfExtents=[0.5, 0.5, 1.5])
        for j in [-4, -2, 0, 2, 4]:
            add_box([9, j, 1.5], halfExtents=[0.5, 0.5, 1.5])
        for i in range(10, 12):
            for j in range(-4, 5):
                add_box([i, j, 2], halfExtents=[0.1, 0.1, 2])
        for i in range(12, 14):
            for j in range(-4, 4):
                add_box([i, j, 2], halfExtents=[0.1, 0.1, 2])

    def surr_1(self):
        # base_pos = [-4.9, -4.9, 0.5], target_pos = [4.5, 4.5, 3.5]
        add_fence(center_pos=[0, 0, 2], internal_length=20.4, internal_width=20.4, height=4, thickness=1)
        for i in range(-19, 19, 1):
            if i == -19:
                j_index = range(-3, 5, 1)
            elif i == 19:
                j_index = range(-4, 4, 1)
            else:
                j_index = range(-4, 5, 1)
            for j in j_index:
                rand = np.random.rand()
                if rand > 0.5:
                    x = 0.1
                    y = 0.1
                    half_height = 2
                    add_box([i, j, half_height], halfExtents=[x, y, half_height])
                elif rand > -1:
                    radius = 0.15
                    height = 4
                    add_cylinder(pos=[i, j, height / 2], radius=radius, height=height)

    def surr_2(self):
        # base_pos = [-4.5, -4.5, 0.5], target_pos = [4.5, 4.5, 3.5]
        for i in range(-4, 5, 2):
            if i == -4 or i==0 or i==4:
                j_index = range(-3, 5, 2)
            else:
                j_index = range(-4, 5, 2)
            for j in j_index:
                x = 0.5
                y = 0.5
                half_height = 2
                add_box([i, j, half_height], halfExtents=[x, y, half_height])

    def surr_3(self):
        # env.reset(base_pos=[-4.5, -4.5, 1], target_pos=[4.7, 4.7, 1.5])
        add_fence(center_pos=[0, 0, 0.75], internal_length=7.2, internal_width=7.2, height=1.5, thickness=0.25)
        # x+ y+
        add_box([3, 3, 1.3], halfExtents=[0.1, 0.1, 1.3])
        add_box([1, 3, 1.3], halfExtents=[0.1, 0.1, 1.3])
        add_box([2, 3, 3], halfExtents=[1.1, 0.1, 0.4])

        add_box([3, 2, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([1, 2, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([2, 2, 1.9], halfExtents=[1.1, 0.1, 0.5])

        add_box([2, 1, 0.6], halfExtents=[1.3, 0.1, 0.6])

        add_box([0, 3, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 0.2, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 1.6, 3], halfExtents=[0.1, 1.5, 0.6])
        # add_box([0, 0, 0.6], halfExtents=[0.1, 3, 0.6])

        # x- y-
        add_box([-3, -3, 1.3], halfExtents=[0.1, 0.1, 1.3])
        add_box([-3, -1, 1.3], halfExtents=[0.1, 0.1, 1.3])
        add_box([-3, -2, 3], halfExtents=[0.1, 1.1, 0.4])

        add_box([-2, -3, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([-2, -1, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([-2, -2, 1.9], halfExtents=[0.1, 1.1, 0.5])

        add_box([-1, -2, 0.6], halfExtents=[0.1, 1.1, 0.6])

        add_box([-3, 0, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 0, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([-1.5, 0, 3], halfExtents=[1.6, 0.1, 0.6])

        add_box([-2, 1, 0.9], halfExtents=[1.1, 0.1, 0.9])

        for i in [-1, -2, -3]:
            for j in [2, 3]:
                add_box([i, j, 2], [0.1, 0.1, 2])

        add_box([1.5, -1.5, 2], [1.3, 1.3, 2], rgba=[0.3, 0.3, 0.3, 0.01])
        # add_box([3, 1, 1], halfExtents=[0.3, 0.3, 1])


def add_box(pos, halfExtents, mass=10000., rgba=[0.3, 0.3, 0.3, 1], physicsClientId=0):
    visual_shape = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=halfExtents,
                                       rgbaColor=rgba,
                                       physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=halfExtents,
                                             physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos,
        physicsClientId=physicsClientId
    )
    return entity_id


def add_fence(center_pos, internal_length, internal_width, height, thickness,
             mass=10000., rgba=[0.7, 0.7, 0.7, 1], physicsClientId=0):
    """
    :param center_pos:      围墙中心的坐标
    :param internal_length: 内部长
    :param internal_width:  内部宽
    :param thickness:       厚度
    :param mass:            质量
    :param rgba:            表面意思
    :return                 四个id，代表组成围墙的四个box的id
    """
    # L1和L2代表长那条线面对面的两面墙，长度为internal_length + 2 * thickness
    L1 = add_box(
        pos=[center_pos[0] + internal_width / 2. + thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    L2 = add_box(
        pos=[center_pos[0] - internal_width / 2. - thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    # W1和W2代表宽那条线面对面的两面墙，长度为internal_length + 2 * thickness
    W1 = add_box(
        pos=[center_pos[0], center_pos[1] + internal_length / 2. + thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    W2 = add_box(
        pos=[center_pos[0], center_pos[1] - internal_length / 2. - thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    return L1, L2, W1, W2


def add_cylinder(pos, radius, height, mass=10000., rgba=[0.3, 0.3, 0.3, 1], physicsClientId=0):
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                       radius=radius,
                                       length=height,
                                       rgbaColor=rgba,
                                       physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                             radius=radius,
                                             height=height,
                                             physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos,
        physicsClientId=physicsClientId
    )
    return entity_id


def add_ball(pos, radius, mass=10000., rgba=[1., 1., 1., 1.], physicsClientId=0):
    visual_shape = p.createVisualShape(p.GEOM_SPHERE,
                                       radius=radius,
                                       rgbaColor=rgba,
                                       physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                             radius=radius,
                                             physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos,
        physicsClientId=physicsClientId
    )
    return entity_id