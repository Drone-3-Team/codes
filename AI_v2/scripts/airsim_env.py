from . import airsim
import gym
import numpy as np
import hp


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done

    def reset(self):
        self.setup_flight()
        obs = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        # Get a random section
        if self.random_start == True:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = section["offset"][0]
        self.target_pos = section["target"]

        # Start the agent at random section at a random yz position
        #y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        #pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        #self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        # Get target distance for reward calculation
        #self.target_dist_prev = np.linalg.norm(np.array([y_pos, z_pos]) - self.target_pos)

    
    def do_action(self, select_action):
        speed = 2
        yaw_speed = 10
        if select_action == 0:
            vx = 0
            vz = speed*0.5
            vyaw = 0
        elif select_action == 1:
            vx = 0
            vz = -speed*0.5
            vyaw = 0
        elif select_action == 2:
            vx = speed
            vz = 0
            vyaw = 0
        elif select_action == 3:
            vx = speed*0.5
            vz = 0
            vyaw = 0
        elif select_action == 4:
            vx = 0
            vz = 0
            vyaw = yaw_speed
        elif select_action == 5:
            vx = 0
            vz = 0
            vyaw = -yaw_speed


        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(vx, 0, vz, duration=1).join()
        self.drone.rotateByYawRateAsync(vyaw, duration = 1).join()

        # # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
        self.drone.rotateByYawRateAsync(yaw_rate = 0, duration=0.1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        x,y,z =  self.drone.simGetVehiclePose().position
        w_val,x_val,y_val,z_val = self.drone.simGetVehiclePose().orientation
        x -= hp.tar_pos[0]
        y -= hp.tar_pos[1]
        z -= hp.tar_pos[2]
        return obs,[x,y,z,w_val,x_val,y_val,z_val]

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
        elif(self.is_collision() or z>450):
            done = 1
            reward = -100
        else:
            reward = -dis

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
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

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        super(TestEnv, self).setup_flight()
        self.eps_n += 1

        # Start the agent at a random yz position
        y_pos, z_pos = (0,0)
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
    def compute_reward(self):
        reward = 0
        done = 0

        x,_,_ = self.drone.simGetVehiclePose().position

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)
    
        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Flight distance (mean): %.2f" % (np.mean(self.agent_traveled)))
            print("> Holes reached (max):", int(np.max(self.agent_traveled)//4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled)//4))
            print("---------------------------------\n")
        
        return reward, done
