# 这里展示了将RGB图片处理成灰度图堆叠成时间序列并喂给cov3d的过程
# cov3d 官方文档:https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d
from inspect import stack
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

#一个只含一层卷积的模型
class COV_LAYER(nn.Module):
    def __init__(self):
        super(COV_LAYER, self).__init__()
        self.layer = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=1,
                                stride=1)
    def forward(self,x):
        return self.layer(x)

# 用于将RGB图片的array转为一个归一化后的灰度图tensor
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
def addDimForCNN(imgStack):
    tensor5D = torch.zeros(1,1,STACK_HEIGHT,IMG_H,IMG_W)
    tensor5D[0,0] = imgStack
    return tensor5D

# 统一调用上面的函数，接受原图像栈、新图片的输入，返回输入CNN的输出以及新的imgStack
def getCNNInput(imgStack,new_RGBimg):
    newImg = img2tensor(new_RGBimg)
    newStack = stack_image(imgStack,newImg)
    return newStack,addDimForCNN(newStack)

# 下面开始运行，创建随机输入

imgStack = torch.randn([STACK_HEIGHT,IMG_H,IMG_W])
newImg = torch.zeros([IMG_H,IMG_W,3])

imgStack,CNNInput = getCNNInput(imgStack,newImg)
print(CNNInput)

l = COV_LAYER()
print(l(CNNInput))