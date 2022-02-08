import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# 超参数们
BATCH_SIZE = 256
LR = 0.05
GAMMA = 0.9
EPISILO = 0.9
MEMORY_CAPACITY = 256
Q_NETWORK_ITERATION = 100
MAX_ROUND = 500

STACK_HEIGHT = 5
IMG_H = 400
IMG_W = 600

# 图像预处理

def img2tensor(rgbArray):
    # 将RGB形式的array[400像素*600像素*3个色彩通道]转为灰度形式array[400像素*600像素]并将像素值从
    # [0,255]归一化到[0,1]
    grayscale_array = np.zeros([IMG_H,IMG_W],dtype=np.double)
    grayscale_array = np.add(grayscale_array,rgbArray[:,:,0])
    grayscale_array = np.add(grayscale_array,rgbArray[:,:,1])
    grayscale_array = np.add(grayscale_array,rgbArray[:,:,2])
    grayscale_array /= 3*255
    #将array转为torch.tensor
    return torch.tensor(grayscale_array,dtype=torch.double)

def stack_image(stack,newImage):
    newImage = torch.unsqueeze(newImage, dim = 0)    # 添加一个维度以使用cat
    stacked = torch.cat([stack[1:],newImage],dim = 0)   #拼接，去除stack[1,:,:]之前的元素
    return stacked

def addDimForCNN(imgStack):
    tensor5D = torch.zeros(1,1,STACK_HEIGHT,IMG_H,IMG_W)
    tensor5D[0,0] = imgStack
    return tensor5D

def getNNInput(imgStack,new_RGBimg):
    newImg = img2tensor(new_RGBimg)
    newStack = stack_image(imgStack,newImg)
    return newStack,addDimForCNN(newStack)

# RGB转深度的网络

# 行动网络

