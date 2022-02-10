import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
from networks import DQN
import hp
from hp import getCNNInput

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
                temp = self.imgStack
                self.imgStack,self.CNNinput = getCNNInput(self.imgStack,state)
                action = self.dqn.predict(self.CNNinput)

                next_state, reward , done, _ = self.trainEnv.step(action)
                _,next_input = getCNNInput(temp,next_state)
                self.dqn.storeTransation(self.CNNinput, action, reward, next_input)
                ep_reward += reward

                if self.dqn.memCnt >= hp.MEMORY_CAPACITY:
                    self.dqn.actor_learn()
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                if done or round_count>hp.MAX_ROUND:
                    break
                state = next_state

            reward_list.append(reward)
            ax.set_xlim(0,300)
            ax.plot(reward_list, 'g-', label='total_loss')
            plt.pause(0.001)
        
    def predict(self):
        return self.dqn.predict(self.CNNinput)

    def save(self):
        self.dqn.save(hp.path+'text_model.pth')

    def load(self,path):
        self.dqn.eval_net = self.dqn.target_net = torch.load(path)
