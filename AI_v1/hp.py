import random
from turtle import shape
from matplotlib import pyplot as plt
import yaml
import scripts.airsim_env as env
import torch
import numpy as np
# 超参数们
BATCH_SIZE = 128
LR = 0.1
GAMMA = 0.9
EPISILO = 0.65
MEMORY_CAPACITY = 256
Q_NETWORK_ITERATION = 128
MAX_ROUND = 1000

EPISODES = 1000
IN_DEPTH = 1
IMG_H = 50
IMG_W = 50

path = './'
ip = '127.0.0.1'

def RGB2Gray(RGBArray):
    gray_array = np.zeros(shape = (IMG_H,IMG_W),dtype=float)
    gray_array += RGBArray[:,:,0]
    gray_array += RGBArray[:,:,1]
    gray_array += RGBArray[:,:,2]
    gray_array = gray_array/(256*3)
    return gray_array


class ExpReplay:
    def __init__(self,device) -> None:
        self.stateMem = np.zeros(shape=(MEMORY_CAPACITY+1,IN_DEPTH,IMG_H,IMG_W),dtype=float)
        self.action = np.zeros(shape=(MEMORY_CAPACITY+1),dtype=int)
        self.reward = np.zeros(shape=(MEMORY_CAPACITY+1),dtype=int)
        self.memCnt = 0
        self.device = device
        
    def push(self,states):
        index = self.memCnt % (MEMORY_CAPACITY+1)
        self.stateMem[index], self.action[index], self.reward[index] = states
        self.memCnt += 1
        
    def replay(self):
        '''
        batchState,batchNexState,batchReawrd,batchAction
        '''
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        batchState = torch.tensor(data = self.stateMem[sample_index,:]
                    ,device=self.device,dtype=torch.float).unsqueeze(dim = 1)
        batchNexState = torch.tensor(data = self.stateMem[sample_index+1,:]
                    ,device=self.device,dtype=torch.float).unsqueeze(dim = 1)
        batchReawrd = torch.tensor(self.reward[sample_index]
                    ,device=self.device, dtype=torch.long)
        batchAction = torch.tensor(np.array([self.action[sample_index]]).astype(int)
                    ,device=self.device,dtype=torch.long)

        '''
        plt.ion()
        plt.subplot(2,1,1)
        plt.imshow(self.stateMem[sample_index[1]][0])
        plt.subplot(2,1,2)
        plt.imshow(self.stateMem[sample_index[1]+1][0])
        plt.pause(0.1)'''

        return batchState,batchNexState,batchReawrd,batchAction