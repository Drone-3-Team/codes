import yaml
import scripts.airsim_env as env

# 超参数们
BATCH_SIZE = 256
LR = 0.05
GAMMA = 0.9
EPISILO = 0.9
MEMORY_CAPACITY = 256
Q_NETWORK_ITERATION = 100
MAX_ROUND = 500
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


EPISODES = 400
STACK_HEIGHT = 5
IMG_H = 400
IMG_W = 600

path = './'
ip = '127.0.0.1'


# 初始化用于训练的环境
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)
trainEnv = env.AirSimDroneEnv(ip,(IMG_H,IMG_W),env_config)
