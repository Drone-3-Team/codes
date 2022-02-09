import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
from networks import DQN
import hp

def img2tensor(depthArray):
    return torch.tensor(depthArray,dtype=torch.double)

def stack_image(stack,newImg):
    newImg = torch.unsqueeze(newImg, dim = 0)    # 添加一个维度以使用cat
    stacked = torch.cat([stack[1:],newImg],dim = 0)   #拼接，去除stack[1,:,:]之前的元素
    return stacked

def addDimForCNN(imgStack):
    tensor5D = torch.zeros(1,1,hp.STACK_HEIGHT,hp.IMG_H,hp.IMG_W)
    tensor5D[0,0] = imgStack
    return tensor5D

def getCNNInput(imgStack,newImg):
    newImg = img2tensor(newImg)
    newStack = stack_image(imgStack,newImg)
    return newStack,addDimForCNN(newStack)


class Agent():
    '''
    用于管理DQN训练
    TODO: 训练多个智能体,从中选优
    '''
    def __init__(self,trainEnv,args) -> None:
        self.dqn = DQN(args)
        self.trainEnv = trainEnv
        self.imgStack = torch.zeros([hp.STACK_HEIGHT,hp.IMG_H,hp.IMG_W],dtype=torch.float)
        self.CNNinput = 0
    def learn(self):
        reward_list = []
        plt.ion()
        fig, ax = plt.subplots()
        for i in range(hp.EPISODES):
            round_count = 0
            state = self.trainEnv.reset()
            ep_reward = 0
            while True:
                round_count += 1

                self.getObs()
                #self.dqn.visiual_learn(state)
                action = self.dqn.predict(self.CNNinput)
                #print(action)
                next_state, _ , done, info = self.trainEnv.step(action)
                self.trainEnv.do_action(action)
                
                reward,_ = self.trainEnv.compute_reward()

                self.dqn.storeTransation(state, action, reward, next_state)
                ep_reward += reward

                if self.dqn.memCnt >= hp.MEMORY_CAPACITY:
                    self.dqn.actor_learn()
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                if done or round_count>hp.MAX_ROUND:
                    break
                state = next_state

            r = copy.copy(reward)
            reward_list.append(r)
            ax.set_xlim(0,300)
            ax.plot(reward_list, 'g-', label='total_loss')
            plt.pause(0.001)

    def getObs(self):
        Img ,_ = self.trainEnv.get_obs()
        #statep = self.dqn.getDepth(Img)
        self.imgStack,self.CNNinput = getCNNInput(self.imgStack,Img)

    def predict(self):
        return self.dqn.predict(self.CNNinput)

    def save(self):
        self.dqn.save(hp.path+'text_model.pth')

    def load(self,path):
        self.dqn.eval_net = self.dqn.target_net = torch.load(path)
