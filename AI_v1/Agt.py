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
            self.inputQue.append(np.zeros(shape=(hp.IMG_H,hp.IMG_W),dtype=float))

        plt.ion()
        self.fig, self.ax = plt.subplots()
        print('using device:', self.device)
    def learn(self):
        reward_list = []
        for i in range(hp.EPISODES):
            round_count = 0
            state = self.trainEnv.reset()

            while True:
                round_count += 1
                self.pushQueue(hp.RGB2Gray(state))
                action = self.dqn.predict(self.getInput())

                next_state, reward , done, _ = self.trainEnv.step(action)
                self.replayMem.push(np.array(self.inputQue), action, reward)

                if self.memCnt >= hp.MEMORY_CAPACITY:
                    self.dqn.actor_learn(self.replayMem.replay())
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, round(reward, 3)))
                if done or round_count>hp.MAX_ROUND:
                    break
                state = next_state

            reward_list.append(reward)
            

    def pushQueue(self,newImg):
        self.inputQue = self.inputQue[1:]
        self.inputQue.append(newImg)

    def getInput(self):
        return torch.tensor(data = np.array(self.inputQue,dtype=float)
                ,device=self.device,dtype=torch.float).unsqueeze(dim = 3)

    def graphing(self,reward_list):
        self.ax.set_xlim(0,300)
        self.ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)

    def save(self):
        self.dqn.save(hp.path+'text_model.pth')

    def load(self,path):
        self.dqn.eval_net = self.dqn.target_net = torch.load(path)