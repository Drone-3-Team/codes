import hp
import torch
import matplotlib.pyplot as plt 
import scripts.airsim_env as env
from Agt import Agent
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
#device = "cpu"  debug

with open('AI_v1\scripts\config.yml', 'r') as f:
    env_config = yaml.safe_load(f)
trainEnv = env.AirSimDroneEnv(hp.ip,(hp.IMG_H,hp.IMG_W,3),env_config['TrainEnv'])

NUM_ACTIONS = trainEnv.action_space.n
NUM_STATES = trainEnv.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(trainEnv.action_space.sample(), int) else trainEnv.action_space.sample.shape

args = [NUM_ACTIONS,NUM_STATES,ENV_A_SHAPE,device]
agt = Agent(trainEnv,args)

agt.learn()
agt.save(hp.path)
