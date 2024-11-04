import airsim
import os
import cv2
import json
import numpy as np
import pandas as pd
from airsim_utils.obs_collector import CollectObservation
from airsim_utils.coords_conversion import *
# [7680.60000013, -3642.26568555, 7.5]

def test_3d_bbox():
    file_path = "E:\\ZWC\\BuildingBBox.json"
    with open(file_path, 'rb') as f:
        data = json.load(f)

    bbox = data["宜盛-配电房4-墙体_2"]
    print(bbox)
    p1 = bbox["p1"]
    p2 = bbox["p2"]

    p1 = np.array([p1['x'], p1['y'], p1['z']])
    p2 = np.array([p2['x'], p2['y'], p2['z']])

    min_point = np.minimum(p1, p2)
    max_point = np.maximum(p1, p2)
    print(min_point, max_point)

    bbox = np.array([
        [min_point[0], min_point[1], min_point[2]],  # P1
        [max_point[0], min_point[1], min_point[2]],  # P2
        [min_point[0], max_point[1], min_point[2]],  # P3
        [max_point[0], max_point[1], min_point[2]],  # P4
        [min_point[0], min_point[1], max_point[2]],  # P5
        [max_point[0], min_point[1], max_point[2]],  # P6
        [min_point[0], max_point[1], max_point[2]],  # P7
        [max_point[0], max_point[1], max_point[2]]  # P8
    ])

    print(bbox.shape)
    bbox_coords = ue_world2airsim_world(bbox)
    bbox_coords = airsim_world2airsim_ego(
        bbox_coords,
        np.array([7680.60000013, -3642.26568555, -7.5]),
        np.array([-0.0, 0.0, 0.0, 1])
    )
    bbox_coords = airsim_ego2camera(bbox_coords)
    bbox_coords = camera2image_coords(bbox_coords, get_intrinsic_matrix(512, 512, 90))
    print(bbox_coords)

    min_point = np.min(bbox_coords, axis=0)
    max_point = np.max(bbox_coords, axis=0)
    print(min_point, max_point)

    img = cv2.imread("./assets/rgb/RGBVis_0.png")
    cv2.rectangle(img, (int(min_point[0]), int(min_point[1])), (int(max_point[0]), int(max_point[1])), (0, 255, 0), 3)
    # for i in range(len(bbox_coords)):
    #     cv2.circle(img, (int(bbox_coords[i][0]), int(bbox_coords[i][1])), 2, (0, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == '__main__':
    # collector = CollectObservation()
    # collector.collect_image([7680.60000013, -3642.26568555, -7.5, 0, 0, 0], "./assets")
    test_3d_bbox()
