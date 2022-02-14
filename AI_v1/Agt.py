import torch
import matplotlib.pyplot as plt
import numpy as np
import airsim
from networks import DuelingDQN
import hp

class Agent():

    def __init__(self,trainEnv,args) -> None:
        self.dqn = DuelingDQN(args)
        self.trainEnv = trainEnv
        self.device = args[3]
        self.replayMem = hp.ExpReplay(self.device)
        self.memCnt = 0
        self.inputQue = []
        for i in range(0,hp.IN_DEPTH):
            self.inputQue.append(np.zeros(shape=(hp.IMG_H,hp.IMG_W),dtype=np.float32))

        #plt.ion()
        #self.fig, self.ax = plt.subplots()
        print('using device:', self.device)
    def learn(self):
        reward_list = []
        for i in range(hp.EPISODES):
            round_count = 0
            state = self.trainEnv.reset()
            tot_reward = 0
            for j in range(0,hp.IN_DEPTH):
                self.inputQue[j] = np.zeros(shape=(hp.IMG_H,hp.IMG_W),dtype=np.float32)
            while True:
                
                round_count += 1
                self.pushQueue(hp.RGB2Gray(state))
                action = self.dqn.predict(self.getInput())

                next_state, reward , done = self.trainEnv.step(action)
                tot_reward += reward

                self.replayMem.push([np.array(self.inputQue), action, reward])
                self.memCnt += 1

                if self.memCnt >= hp.MEMORY_CAPACITY:
                    self.dqn.actor_learn(self.replayMem.replay())
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, tot_reward))
                if done or round_count>hp.MAX_ROUND:
                    break
                state = next_state

            reward_list.append(tot_reward)
            #self.graphing(reward_list,i)
            

    def pushQueue(self,newImg):
        self.inputQue = self.inputQue[1:]
        self.inputQue.append(newImg)

    def getInput(self):
        return torch.tensor(data = np.array(self.inputQue,dtype=float)
                ,device=self.device,dtype=torch.float).unsqueeze(dim = 0).unsqueeze(dim = 1)

    def graphing(self,reward_list,i):
        self.ax.set_xlim(0,i)
        self.ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)

    def save(self):
        self.dqn.save(hp.path+'text_model.pth')

    def load(self,path):
        self.dqn.eval_net = self.dqn.target_net = torch.load(path)