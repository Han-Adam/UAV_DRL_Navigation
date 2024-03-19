import pybullet as p
import numpy as np
import time


class Camera:
    def __init__(self, client):
        self.client = client
        self.width = 256
        self.height = 256
        index_map = ((np.array(range(self.width)) + 0.5) / self.width * 2 - 1).tolist()
        self.index_map = np.reshape(index_map * self.height, [self.width, self.height])
        """
        -1, -0.9 ,-0.8, ... ,0.8, 0.9, 1
        -1, -0.9 ,-0.8, ... ,0.8, 0.9, 1
        -1, -0.9 ,-0.8, ... ,0.8, 0.9, 1
        """

    def map_refresh(self, r, g, b, map3d, pos, matrix, path_map):
        print('map refresh')
        tx_vec = matrix[:, 0]
        tz_vec = matrix[:, 2]
        cameraPos = np.array(pos) + tx_vec * 0.03 - tz_vec * 0.035
        targetPos = cameraPos + tx_vec

        depth = self.get_image(cameraPos, targetPos, tz_vec)
        valid_index = np.reshape(depth <= 10, [-1])

        point_cloud = np.empty(shape=[self.width, self.height, 3])
        # 注意世界坐标系与相机坐标系的匹配
        point_cloud[:, :, 0] = depth
        point_cloud[:, :, 1] = - depth * self.index_map
        point_cloud[:, :, 2] = - depth * self.index_map.T

        point_cloud = np.reshape(point_cloud, [-1, 3]) # [valid_index, :]

        point_cloud = np.matmul(matrix, point_cloud.T).T + cameraPos
        point_cloud = point_cloud[valid_index, :]

        voxel_index = np.floor(point_cloud * 5)
        voxel_index = np.array(voxel_index, dtype=np.compat.long)
        # voxel_index = np.unique(voxel_index, axis=0)
        # voxel_index = np.concatenate([voxel_index,
        #                               voxel_index + [1, 0, 0],
        #                               voxel_index - [1, 0, 0],
        #                               voxel_index + [0, 1, 0],
        #                               voxel_index - [0, 1, 0],
        #                               voxel_index + [1, 1, 0],
        #                               voxel_index - [1, 1, 0],
        #                               voxel_index + [1, -1, 0],
        #                               voxel_index - [-1, 1, 0]])
        voxel_index = np.unique(voxel_index, axis=0) + [75, 25, 0] + [20, 20, 20]
        # print(voxel_index.shape)
        # for i in range(voxel_index.shape[0]):
        #     print(voxel_index[i, :])
        # time.sleep(1000)



        map3d[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1

        # for i in voxel_index.shape[0]:
        #     print(voxel_index[i])
        # time.sleep(1000)

        if np.any(path_map[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] == 1):
            return True
        return False

    def get_image(self, cameraPos, targetPos, tz_vec):
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec,
            physicsClientId=self.client
        )
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=90,  # 摄像头的视线夹角
            aspect=1.0,  # 屏幕宽高比
            nearVal=0.01,  # 摄像头焦距下限
            farVal=20,  # 摄像头能看上限
            physicsClientId=self.client
        )
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            physicsClientId=self.client
        )
        depthImg = 20*0.01/(20 - (20-0.01)*depthImg)
        return depthImg
