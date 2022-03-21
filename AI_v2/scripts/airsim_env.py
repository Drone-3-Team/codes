from . import airsim
import gym
import numpy as np
import hp


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)#图像通道
        self.action_space = gym.spaces.Discrete(6)

        self.collision_time = 0
        self.random_start = True
        self.active_bomb = False
        self.setup_flight()

        self.spd  = 0
        self.yawspd = 0

    def step(self, action):
        self.do_action(action)
        obs = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done

    def reset(self):
        if self.active_bomb:
            self.bomb()
        self.active_bomb = False
        self.setup_flight()
        obs = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # 无控制是悬停
        self.drone.moveToZAsync(-1, 1)

        # 获取碰撞
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
    
    # 根据智能体作出的选择，修改airsim环境中的飞机速度
    def do_action(self, select_action):
        if select_action == 0:
            dv = 0
            dvyaw = 0
        elif select_action == 1:
            dv = 0.1
            dvyaw = 0
        elif select_action == 2:
            dv = -0.1
            dvyaw = 0
        elif select_action == 3:
            dv = 0
            dvyaw = 10
        elif select_action == 4:
            dv = 0
            dvyaw = -10
        elif select_action == 5:
            dv = -self.spd
            dvyaw = -self.yawspd

        self.spd += dv
        self.yawspd += dvyaw

        # 执行
        self.drone.moveByVelocityBodyFrameAsync(dv, 0, self.spd, duration=1).join()
        self.drone.rotateByYawRateAsync(self.yawspd, duration = 1).join()

    # 获取观测信息。观测信息包含：图像、与目标的相对位置
    def get_obs(self):
        obs = self.get_rgb_image()
        x,y,z =  self.drone.simGetVehiclePose().position
        w_val,x_val,y_val,z_val = self.drone.simGetVehiclePose().orientation
        x -= hp.tar_pos[0]
        y -= hp.tar_pos[1]
        z -= hp.tar_pos[2]
        return obs,[x,y,z,w_val,x_val,y_val,z_val]

    # 计算奖励函数。奖励函数原则：当抵达20距离内成功得100分，当坠机、被发现失败得-100分，其他情况下得1000/距离 分
    def compute_reward(self):
        reward = 0
        done = 0

        x,y,z =  self.drone.simGetVehiclePose().position
        x -= hp.tar_pos[0]
        y -= hp.tar_pos[1]
        z -= hp.tar_pos[2]

        dis = np.sqrt(x**2 + y**2 + z**2)

        if(dis < 20):
            done = 1
            reward = 100
        elif(self.is_collision() or self.is_captured()):
            done = 1
            reward = -100
        else:
            reward = 1000/dis

        return reward, done

    # 坠机
    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False

    #被捕
    def is_captured(self):
        # 记得active——bomb
        pass

    def bomb():
        pass
    
    #通过airsim获取RGB观测
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

    #通过airsim获取深度图
    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image

