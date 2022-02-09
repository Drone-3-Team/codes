import hp
import copy
import torch
import matplotlib.pyplot as plt
import scripts.airsim_env as env
from networks import DQN
import yaml

# 初始化用于训练的环境
with open('AI_v1\scripts\config.yml', 'r') as f:
    env_config = yaml.safe_load(f)
trainEnv = env.AirSimDroneEnv(hp.ip,(hp.IMG_H,hp.IMG_W),env_config['TrainEnv'])

NUM_ACTIONS = trainEnv.action_space.n
NUM_STATES = trainEnv.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(trainEnv.action_space.sample(), int) else trainEnv.action_space.sample.shape
'''T1
测试深度图获取OK
Img,collision = trainEnv.get_obs()
plt.imshow(Img)
plt.show()
'''
