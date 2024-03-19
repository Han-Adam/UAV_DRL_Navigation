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
        add_fence(center_pos=[0, 0, 2], internal_length=10.4, internal_width=10.4, height=4, thickness=1)
        self.surr_3()
        p.setGravity(0., 0., -self.G)
        p.setTimeStep(self.time_step)

    def surr_1(self):
        # base_pos = [-4.9, -4.9, 0.5], target_pos = [4.5, 4.5, 3.5]
        for i in range(-4, 5, 1):
            if i == -4:
                j_index = range(-3, 5, 1)
            elif i == 4:
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
                x = 0.5 # np.random.rand()*0.4 + 0.3 # 0.5
                y = 0.5 # np.random.rand()*0.4 + 0.3 # 0.5
                half_height =  2 # np.random.rand()*1.8+0.2 # 2
                add_box([i, j, half_height], halfExtents=[x, y, half_height])

    def surr_3(self):
        add_fence(center_pos=[0, 0, 0.75], internal_length=7.1, internal_width=7.3,
                  height=1.5, thickness=0.2, rgba=[0.3, 0.3, 0.3, 1])
        # x+ y+
        add_box([3, 3, 1.1], halfExtents=[0.1, 0.1, 1.1])
        add_box([1, 3, 1.1], halfExtents=[0.1, 0.1, 1.1])
        add_box([2, 3, 2.7], halfExtents=[1.1, 0.1, 0.5])

        add_box([3, 2, 0.5], halfExtents=[0.1, 0.1, 0.5])
        add_box([1, 2, 0.5], halfExtents=[0.1, 0.1, 0.5])
        add_box([2, 2, 1.7], halfExtents=[1.1, 0.1, 0.7])

        add_box([2, 1, 0.5], halfExtents=[1.3, 0.1, 0.5])

        add_box([0, 3, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 0.2, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 1.6, 3], halfExtents=[0.1, 1.5, 0.6])

        # x- y-
        add_box([-3, -3, 1.3], halfExtents=[0.1, 0.2, 1.3])
        add_box([-3, -1, 1.3], halfExtents=[0.1, 0.2, 1.3])
        add_box([-3, -2, 3], halfExtents=[0.1, 1.2, 0.4])

        add_box([-2, -3, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([-2, -1, 0.7], halfExtents=[0.1, 0.1, 0.7])
        add_box([-2, -2, 1.9], halfExtents=[0.1, 1.1, 0.5])

        add_box([-1, -2, 0.7], halfExtents=[0.1, 1.1, 0.7])

        add_box([-3, 0, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([0, 0, 1.2], halfExtents=[0.1, 0.1, 1.2])
        add_box([-1.5, 0, 3], halfExtents=[1.6, 0.1, 0.6])
        add_box([-1.5, 0, 0.75], halfExtents=[1.4, 0.1, 0.75])


        add_box([-2, 1, 0.8], halfExtents=[1.1, 0.1, 0.8])

        for i in [-1, -2, -3]:
            for j in [2, 3]:
                add_box([i, j, 1], [0.1, 0.1, 1])

        add_box([1.5, -1.5, 2], [1.6, 1.5, 2], rgba=[0.3, 0.3, 0.3, 0.01])


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