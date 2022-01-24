# 这是一个用来临时运行一些东西的py文件
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

STACK_HEIGHT = 5
IMG_H = 4
IMG_W = 6

class COV_LAYER(nn.Module):
    def __init__(self):
        super(COV_LAYER, self).__init__()
        self.layer = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=(STACK_HEIGHT,1,1))
    def forward(self,x):
        return self.layer(x)

# 用于将RGB图片的array转为一个归一化后的灰度图tensor
def img2tensor(rgbArray):
    # 将RGB形式的array[400像素*600像素*3个色彩通道]转为灰度形式array[400像素*600像素]并将像素值从
    # [0,255]归一化到[0,1]
    grayscale_array = np.zeros([IMG_H,IMG_W],dtype=np.double)
    grayscale_array += rgbArray[:,:,0]
    grayscale_array += rgbArray[:,:,1]
    grayscale_array += rgbArray[:,:,2]
    grayscale_array /= 3*255
    #将array转为torch.tensor
    return torch.tensor(grayscale_array,dtype=torch.double)

# 用于将图像新的一帧加到stack上并去除最旧的帧
def stack_image(stack,newImage):
    newImage = torch.unsqueeze(newImage, dim = 0)    # 添加一个维度以使用cat
    stacked = torch.cat([stack[1:],newImage],dim = 0)   #拼接，去除stack[1,:,:]之前的元素
    return stacked

# cov3d的输入tensor长相为(N,Cin,D,H,W)
# N为样本数（一般为1）
# Cin为输入通道数，例如RGB为3，灰阶为1
# D时间序列的深度，这个cov2d没有
# H,W 图像尺寸
# 因此需要将三维的灰度图栈tensor转为五维的tensor
def addDimForCorefunc(imgStack):
    imgStack = torch.unsqueeze(imgStack,dim = 0)
    imgStack = torch.unsqueeze(imgStack,dim = 0)
    return imgStack

old = torch.randn([5,IMG_H,IMG_W])
new = torch.zeros([IMG_H,IMG_W])

l = COV_LAYER()
result = l.forward()

print(stacked[1:])