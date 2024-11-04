# Author: Weichen Zhang
# 该文档代码参考 https://hxy78x0w82.feishu.cn/wiki/NI1swal55iOI9Rkjy5AckjH8nnf 实现不同坐标系之间的互相转换

from scipy.spatial.transform import Rotation as R
import airsim
import numpy as np


def get_intrinsic_matrix(height, width, fov):
    fov_x = np.radians(fov)
    fov_y = 2 * np.arctan((height*1.0 / width) * np.tan(fov_x / 2))

    intrinsic_parameters = {
        'width': width,
        'height': height,
        'fx': width / (2 * np.tan(fov_x / 2)), # 1.5 * width,
        'fy': height / (2 * np.tan(fov_y / 2)), # 1.5 * width,
        'cx': width / 2,
        'cy': height / 2,
    }
    intrinsic_matrix = np.eye(3)
    intrinsic_matrix[0, 0] = intrinsic_parameters['fx']
    intrinsic_matrix[1, 1] = intrinsic_parameters['fy']
    intrinsic_matrix[0, 2] = intrinsic_parameters['cx']
    intrinsic_matrix[1, 2] = intrinsic_parameters['cy']

    return intrinsic_matrix


# xyzw to roll, pitch, yaw
def quaternion2eularian_angles(quat):
    pry = airsim.to_eularian_angles(quat)    # p, r, y
    return np.array([pry[1], pry[0], pry[2]])


def quaternion2np_quaternion(quat):
    return np.array([quat.w_val, quat.x_val, quat.y_val, quat.z_val])
def ue_world2airsim_world(ue_coords):
    '''
    Args:
        ue_coords: (n, 3) numpy array in UE world coordinates

    Returns:
        airsim_world: (n, 3) numpy array in Airsim world coordinates
    '''
    airsim_world = np.hstack((ue_coords[:, 0:1]/100, ue_coords[:, 1:2]/100, -ue_coords[:, 2:3]/100))
    return airsim_world


def airsim_world2airsim_ego(world_coords, pos, ori):
    '''
    Convert points in airsim world coordinates to agent ego coordinates system
    Args:
        world_coords: (n, 3) numpy array in Airsim world coordinates
        pos: agent position: [x, y, z] numpy array
        ori: agent orientation, [x, y, z, w] numpy array
    '''
    world_homo = np.hstack((world_coords, np.ones((len(world_coords), 1))))
    r1 = R.from_quat(ori).as_matrix()
    r2 = R.from_euler('x', 180, degrees=True).as_matrix()
    rot = r1.dot(r2)

    ego_coords = (world_coords - pos).dot(rot)
    return ego_coords


def airsim_ego2camera(ego_coords):
    '''
    Convert ego coordinates to virtual camera coordinates
    Args:
        ego_coords: (n, 3) numpy array
    Returns:
    '''
    X = -ego_coords[:, 1:2]
    Y = -ego_coords[:, 2:3]
    Z = ego_coords[:, 0:1]

    camera_coords = np.hstack((X, Y, Z))
    return camera_coords


def camera2image_coords(camera_coords, intrinsic):
    '''
    Convert camera coordinates to image
    Args:
        camera_coords: (n, 3) numpy array
        intrinsic: (3, 3) numpy array
    Returns:
    '''

    X = camera_coords[:, 0:1]
    Y = camera_coords[:, 1:2]
    Z = camera_coords[:, 2:3]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    xv = X / Z * fx + cx
    yv = Y / Z * fy + cy

    image_coords = np.hstack((xv, yv))
    return image_coords


def image_coords2camera(depth_map, intrinsic):
    height, width = depth_map.shape[:2]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)

    xv, yv = np.meshgrid(x, y)

    Z = depth_map
    X = (xv - cx) * Z / fx
    Y = (yv - cy) * Z / fy

    camera_coords = np.hstack((X, Y, Z))
    return camera_coords


def camera2airsim_ego(camera_coords):
    ego_coords = np.hstack((camera_coords[:, 2], -camera_coords[:, 0], -camera_coords[: 1]))

    return ego_coords


def airsim_ego2airsim_world(ego_coords, pos, ori):
    '''
    Convert ego coordinates to airsim world coordinates
    Args:
        ego_coords: (n, 3) numpy array of ego coordinates
        pos: [x, y, z] numpy array
        ori: [x, y, z, w] numpy array

    Returns:
    '''
    ego_homo = np.hstack((ego_coords, np.ones(ego_coords.shape[0])))
    r1 = R.from_quat(ori).as_matrix()
    r2 = R.from_euler('x', 180, degrees=True).as_matrix()
    rot = r1.dot(r2)

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot
    trans_mat[:3, 3] = pos

    words_homo = ego_homo.dot(trans_mat.T)

    return words_homo[:, :3]


def airsim_world2ue_world(airsim_world):
    ue_worlds = np.hstack((airsim_world[:, 0]*100, airsim_world[:, 1]*100, -airsim_world[:, 2]*100))

    return ue_worlds
