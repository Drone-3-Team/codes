import setup_path
import airsim
import numpy as np
import math
import time
from PIL import Image
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(84,84,1),time_scale=1e-1):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0
        self.simulation_time_scale = time_scale # 仿真步长，也就是每个动作执行多长时间

        self.state = {
            "pose": np.zeros(3),
            "collision": False,
            "prev_pose": np.zeros(3),
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)

        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False) # 浮点图像，不压缩
        # airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)
        # airsim.ImageType.DepthVis
        # airsim.ImageType.Scene
        # airsim.ImageType.Segmentation
        # airsim.ImageType.SurfaceNormals
        # airsim.ImageType.Infrared
        # airsim.ImageType.DisparityNormalized

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.state["pose"] = None
        self.state["prev_pose"] = None
        self.state["collision"] = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True) # 打开API控制使能
        self.car.armDisarm(True) # False貌似是把载具在场景中去掉了
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action, delay=None):
        '''
        action = 0：刹车
        action = 1：直行
        action = 2：大幅度左转
        action = 3：大幅度右转
        action = 4：小幅度左转
        action = 其他值：小幅度右转
        '''
        self.car_controls.brake = 0
        self.car_controls.throttle = 1 # 油门
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5 # 具体表示左转或右转其实不重要，知道有一个表示左转，另一个右转就可以了
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        # 令采取的动作执行一定时间
        if delay is not None:
            time.sleep(delay)
        elif delay is None:
            time.sleep(self.simulation_time_scale)

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize(self.image_shape[0:2]).convert("L"))

        return im_final.reshape(self.image_shape) # im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses)

        self.car_state = self.car.getCarState()
        collision = self.car.simGetCollisionInfo().has_collided

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        '''
        完成路径巡航任务，也就是按规定路线走完全程
        同时要求车速越快越好，且不能撞车
        '''
        MAX_SPEED = 300
        MIN_SPEED = 10
        thresh_dist = 3.5
        beta = 3

        z = 0
        pts = [    # NED坐标系，画了一个“日”字，按顺序访问这些地点
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, 125, z]),
            np.array([0, 125, z]),
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, -128, z]),
            np.array([0, -128, z]),
            np.array([0, -1, z]),
        ]
        pd = self.state["pose"].position
        car_pt = np.array([pd.x_val, pd.y_val, pd.z_val]) # 载具的当前坐标

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-beta * dist) - 0.5
            reward_speed = (                               # 奖励高车速
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed

        done = False
        if reward < -1:
            done = True
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:  # 车速过低也结束交互
                done = True
        if self.state["collision"]:        # 若碰撞，则结束交互
            done = True

        return reward, done

    def step(self, action, dalay):# 加入了delay项，表示该动作持续的时间，可以自由控制仿真的时间颗粒度
        self._do_action(action,delay)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1,0.1)
        return self._get_obs()