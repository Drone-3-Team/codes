import yaml
import scripts.airsim_env as env
import torch

# 超参数们
BATCH_SIZE = 128
LR = 0.05
GAMMA = 0.9
EPISILO = 0.9
MEMORY_CAPACITY = 128
Q_NETWORK_ITERATION = 64
MAX_ROUND = 1000

EPISODES = 10000
STACK_HEIGHT = 5
IMG_H = 50
IMG_W = 50

path = './'
ip = '127.0.0.1'

def img2tensor(depthArray):
    return torch.tensor(depthArray,dtype=torch.double)

def stack_image(stack,newImg):
    newImg = torch.unsqueeze(newImg, dim = 0)    # 添加一个维度以使用cat
    stacked = torch.cat([stack[1:],newImg],dim = 0)   #拼接，去除stack[1,:,:]之前的元素
    return stacked

def addDimForCNN(imgStack):
    tensor5D = torch.zeros(1,1,STACK_HEIGHT,IMG_H,IMG_W)
    tensor5D[0,0] = imgStack
    return tensor5D

def getCNNInput(imgStack,newImg):
    newImg = img2tensor(newImg)
    newStack = stack_image(imgStack,newImg)
    return newStack,addDimForCNN(newStack)
