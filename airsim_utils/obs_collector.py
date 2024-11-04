import airsim
import os
import cv2
import json
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from .coords_conversion import quaternion2eularian_angles, quaternion2np_quaternion


class CollectObservation:
    def __init__(self):
        # 连接到 AirSim
        # client = airsim.MultirotorClient()
        client = airsim.VehicleClient()
        client.confirmConnection()
        # client.enableApiControl(True)
        # client.armDisarm(True)
        #
        # # 起飞
        # client.takeoffAsync().join()

        self.client = client

    def collect_video(self, directory_path):

        video_path = os.path.join(directory_path, "video")
        os.makedirs(video_path, exist_ok=True)

        client = self.client

        # 设定旋转速度和总旋转角度
        yaw_rate = 30  # 每秒旋转30度
        total_rotation = 360  # 旋转一圈
        rotation_duration = total_rotation / yaw_rate  # 旋转一圈所需时间

        # 设置视频保存参数
        frame_width = 1280
        frame_height = 720
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 确保摄像头角度是水平的
        # print("确保摄像头角度是水平的")
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        client.simSetCameraPose("0", camera_pose)

        # 开始旋转并记录视频
        start_time = time.time()
        frame_times = []
        frames = []

        client.rotateByYawRateAsync(yaw_rate, rotation_duration)  # 开始旋转

        while time.time() - start_time < rotation_duration - 0.1:
            frame_start_time = time.time()

            # 获取摄像头图像
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            if img1d.size != (responses[0].height * responses[0].width * 3):
                print("图像大小不匹配，跳过该帧")
                continue

            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

            # 调整图像大小
            img_rgb = cv2.resize(img_rgb, (frame_width, frame_height))

            # 保存图像到帧列表
            frames.append(img_rgb)

            # 记录每帧的时间
            frame_times.append(time.time() - frame_start_time)

        # 停止旋转
        client.rotateByYawRateAsync(0, 1).join()

        # 计算实际帧率
        average_frame_time = sum(frame_times) / len(frame_times)
        actual_fps = 1 / average_frame_time if average_frame_time > 0 else 1

        print(f"实际帧率: {actual_fps:.2f} FPS")

        # 初始化旋转视频写入器
        rotation_video = cv2.VideoWriter(os.path.join(video_path, "drone_rotation.avi"), fourcc, actual_fps, (frame_width, frame_height))

        # 将所有帧写入旋转视频
        for frame in frames:
            rotation_video.write(frame)

        # 确保所有帧都正确写入
        rotation_video.release()

        # 初始化云台移动视频写入器
        gimbal_video = cv2.VideoWriter(os.path.join(video_path, "gimbal_movement.avi"), fourcc, actual_fps, (frame_width, frame_height))

        # 控制云台角度向下到90度，然后向上到-90度，并记录视频
        def record_gimbal_movement(start_angle, end_angle, duration, video_writer, step=1):
            num_steps = int(duration * actual_fps)
            angle_step = (end_angle - start_angle) / num_steps
            for i in range(num_steps):
                angle = start_angle + i * angle_step
                # 将角度转换为四元数设置相机姿态
                camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(angle * np.pi / 180, 0, 0))
                client.simSetCameraPose("0", camera_pose)
                time.sleep(1 / actual_fps)

                # 获取摄像头图像
                responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                if img1d.size != (responses[0].height * responses[0].width * 3):
                    print("图像大小不匹配，跳过该帧")
                    continue

                img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

                # 调整图像大小
                img_rgb = cv2.resize(img_rgb, (frame_width, frame_height))

                # 保存图像到视频
                video_writer.write(img_rgb)

            # 确保最后一帧的角度是 end_angle
            camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(end_angle * np.pi / 180, 0, 0))
            client.simSetCameraPose("0", camera_pose)
            time.sleep(1 / actual_fps)

            # 再次获取摄像头图像
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            if img1d.size == (responses[0].height * responses[0].width * 3):
                img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
                img_rgb = cv2.resize(img_rgb, (frame_width, frame_height))
                video_writer.write(img_rgb)

        # 向下移动到90度
        record_gimbal_movement(0, 90, 3, gimbal_video)  # 3秒钟从0度到90度

        # 向上移动到-90度
        record_gimbal_movement(90, -90, 6, gimbal_video)  # 6秒钟从90度到-90度

        # 恢复云台到水平位置
        record_gimbal_movement(-90, 0, 3, gimbal_video)  # 3秒钟从-90度到0度

        # 确保所有帧都正确写入
        gimbal_video.release()

        # # 降落并解除控制
        # client.landAsync().join()
        # client.armDisarm(False)
        # client.enableApiControl(False)

        print("视频录制完成并保存为 'drone_rotation.avi' 和 'gimbal_movement.avi'")

        # # 使用 moviepy 来合并两个视频文件
        # try:
        #     rotation_clip = mpy.VideoFileClip('drone_rotation.avi')
        #     gimbal_clip = mpy.VideoFileClip('gimbal_movement.avi')
        #     final_clip = mpy.concatenate_videoclips([rotation_clip, gimbal_clip])
        #     final_clip.write_videofile(os.path.join(video_path, "drone_combined.avi"), codec='libx264')
        #     print("视频合并完成并保存为 'drone_combined.avi'")
        # except Exception as e:
        #     print(f"视频合并过程中出错: {e}")

    def collect_image(self, pose, directory_path):
        self.setVehiclePose(pose)
        os.makedirs(directory_path, exist_ok=True)
        self.getPanoState(view_num=12, if_save=True, save_dir=directory_path)
        print(f"sample saved in {directory_path}")

    def setVehiclePose(self, pose: np.ndarray) -> None:
        '''
        pose为[pos, rot]
        rot接受欧拉角或者四元数，
        如果len(pose) == 6,则认为rot为欧拉角,单位为弧度, [pitch, roll, yaw]
        如果len(pose) == 7,则认为rot为四元数, [x, y, z, w]
        '''
        pos = pose[:3]
        rot = pose[3:]

        if len(rot) == 3:
            air_rot = airsim.to_quaternion(rot[0], rot[1], rot[2])
            # print(air_rot)
        elif len(rot) == 4:
            air_rot = airsim.Quaternionr()
            air_rot.x_val = rot[0]
            air_rot.y_val = rot[1]
            air_rot.z_val = rot[2]
            air_rot.w_val = rot[3]
        else:
            raise ValueError(f"Expected rotation shape is (4,) or (3, ), got ({len(rot)},)")

        air_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        air_pose = airsim.Pose(air_pos, air_rot)
        self.client.simSetVehiclePose(air_pose, ignore_collision=True)
        self.gt_height = float(air_pos.z_val)

    def getPanoState(self, view_num=12, if_save=True, save_dir="./assets"):
        rgb_obs = []
        dep_obs = []

        step_angle = 360 // view_num
        rest_angle = 360 - (step_angle - 1) * view_num
        if 360 % view_num != 0:
            print("Warning: view num is not divisible by 360")

        os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "state"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "depth_vis"), exist_ok=True)

        init_pos, init_rot = self.get_current_state(quat=False, degree=False)

        for i in range(view_num):
            state = {}
            # print(i)
            self.makeAction(3, degree=step_angle)
            rgb_img, dep_img = self.getObsRGBD()
            rgb_obs.append(rgb_img)
            dep_obs.append(dep_img)

            pos, ori = self.get_current_state(quat=True)
            state["rotation"] = ori.tolist()
            state["translation"] = pos.tolist()
            # print(ori)

            # visualization
            depth_img_vis = dep_img / 100
            depth_img_vis[depth_img_vis > 1] = 1.
            img_depth_vis = (depth_img_vis * 255).astype(np.uint8)

            # 4. 保存为文件
            if if_save:
                heading = (i+1) * step_angle % 360
                cv2.imwrite(f'{save_dir}/depth_vis/DepthVis_{heading}.png', img_depth_vis)
                cv2.imwrite(f"{save_dir}/rgb/RGBVis_{heading}.png", rgb_img)
                np.save(f"{save_dir}/depth/Depth_{heading}.npy", dep_img)
                state_str = json.dumps(state)
                with open(f"{save_dir}/state/RGBVis_{heading}.json", "w") as f:
                    f.write(state_str)

        # convert back to the initial state
        self.setVehiclePose(np.concatenate([init_pos, init_rot], axis=0))

        return rgb_obs, dep_obs

    def makeAction(self, act_enum, hold=False, **kwargs):
        new_pose = self.getPoseAfterAction(act_enum, **kwargs)
        self.setVehiclePose(new_pose)
        if hold:
            self.client.hoverAsync().join()
        # time.sleep(1)
        # self.client.moveToPositionAsync(float(new_pose[0]), float(new_pose[1]), float(new_pose[2]), 1).join()

    def getPoseAfterAction(self, act_enum, **kwargs):
        cur_pos, cur_rot = self.get_current_state()

        cur_pos[2] = self.gt_height

        new_pos = cur_pos
        new_rot = np.rad2deg(cur_rot)
        # print("new rot:", new_rot)

        if act_enum == 1:           # forward 10 m
            act_step = kwargs['dist'] if 'dist' in kwargs else 10
            new_pos[0] = cur_pos[0] + act_step * np.cos(np.deg2rad(cur_rot[2]))
            new_pos[1] = cur_pos[1] + act_step * np.sin(np.deg2rad(cur_rot[2]))
        elif act_enum == 2:         # turn left by 45 degrees
            act_step = kwargs['degree'] if 'degree' in kwargs else 45
            new_rot[2] = new_rot[2] - act_step if act_step != 0 else new_rot[2]
        elif act_enum == 3:         # turn right by 45 degrees
            act_step = kwargs['degree'] if 'degree' in kwargs else 45
            new_rot[2] = new_rot[2] + act_step if act_step != 0 else new_rot[2]
        elif act_enum == 4:         # go up by 5 m
            act_step = kwargs['dist'] if 'dist' in kwargs else 5
            new_pos[2] = cur_pos[2] - act_step
        elif act_enum == 5:         # go down by 5 m
            act_step = kwargs['dist'] if 'dist' in kwargs else 5
            new_pos[2] = cur_pos[2] + act_step
        else:
            print(f"Unknown action {act_enum}, keep still.")

        new_rot = np.deg2rad(new_rot)

        return np.concatenate((new_pos, new_rot), axis=0)

    def get_current_state(self, quat=False, degree=False):
        # get world frame pos and orientation
        # orientation is in roll, pitch, yaw format
        state = self.client.simGetGroundTruthKinematics()
        pos = state.position.to_numpy_array()
        ori = state.orientation
        if not quat:
            ori = quaternion2eularian_angles(ori)       # [p, r, y]
            if degree:
                ori = np.rad2deg(ori)
        else:
            # print(ori)
            ori = quaternion2np_quaternion(ori)         # [w, x, y, z]

        return pos, ori

    def getObsRGBD(self, camera_id='0'):
        # 请求图像类型
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),  # RGB 图像
            airsim.ImageRequest(camera_id, airsim.ImageType.DepthPlanar, True, False)  # DepthPlanner 图像
        ])

        # 处理 RGB 图像
        rgb_response = responses[0]
        rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)  # 将图像数据转换为 1D 数组
        rgb_img = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)  # 将 1D 数组重塑为 3D 图像数组

        # 处理 DepthPlanner 图像
        depth_response = responses[1]
        depth_img1d = np.array(depth_response.image_data_float, dtype=np.float32)  # 将图像数据转换为 1D 数组
        depth_img = depth_img1d.reshape(depth_response.height, depth_response.width)  # 将 1D 数组重塑为 2D 深度图像数组

        return rgb_img, depth_img

    def run(self):
        dataset_point = pd.read_csv('dataset/dataset_point.csv')
        # for i in range(0, dataset_point.shape[0]):
        for i in range(800, dataset_point.shape[0]):
            pos = dataset_point.iloc[i].values
            pos[3] = 0
            pos[4] = 0
            self.setVehiclePose(pos)
            directory_path = 'dataset/embodied_tasks/%d' % i
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            self.getPanoState(view_num=12, if_save=True, save_dir=directory_path)
            print(f"{i} sample saved in {directory_path}")
