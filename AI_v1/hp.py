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

EPISODES = 400
STACK_HEIGHT = 5
IMG_H = 400
IMG_W = 600

path = './'
ip = '127.0.0.1'