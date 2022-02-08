from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scripts.airsim_env as env

# 超参数们
BATCH_SIZE = 256
LR = 0.05
GAMMA = 0.9
EPISILO = 0.9
MEMORY_CAPACITY = 256
Q_NETWORK_ITERATION = 100
MAX_ROUND = 500
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


STACK_HEIGHT = 5
IMG_H = 400
IMG_W = 600

# 深度预测网络.暂时不使用
class DepthPredict():
    def __init__(self) -> None:
        pass
    def forward(x):
        return x

# 根据深度图决策的行动网络
class Actor_Net(nn.Module):
    def __init__(self):
        super(Actor_Net, self).__init__()
        self.covLayer1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,5,5),stride=(1,5,5)),
            nn.Conv3d(in_channels=1,out_channels=5,kernel_size=(1,5,5),stride=(3,5,5)),
            nn.BatchNorm3d(num_features=5),
        )

        self.covLayer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2)),
            nn.Conv3d(in_channels=5,out_channels=5,kernel_size=(1,2,2),stride=(1,2,2)),
            nn.BatchNorm3d(num_features=5),
        )
        self.fc1 = nn.Linear(233, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.covLayer1(x)
        x = self.covLayer2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

'''
实现的功能:
.load(文件路径)
.predict(当前局面)
.learn(上一轮结果)
.save(文件路径)
'''
class DQN():

    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Actor_Net(), Actor_Net()
        self.visual_net = DepthPredict()

        self.learnStepCNT = 0
        self.memCnt = 0
        self.expMem = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.stateMem = torch.rand(2,dtype=float)

        self.optimizer0 = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.lossFn0 = nn.MSELoss()
        '''
        self.optimizer1 = torch.optim.Adam(self.visual_net.parameters(), lr=LR)
        self.lossFn1 = nn.L1Loss()'''

        self.imgStack = torch.ones([STACK_HEIGHT,IMG_H,IMG_W],dtype=torch.float)


    def predict(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def storeTransation(self,state,action,reward,nextState):
        index = self.memCnt % MEMORY_CAPACITY        
        self.expMem[index,:] = np.hstack((state,[action,reward],nextState))
        self.memCnt += 1

    def visiual_learn(self,envState):
        '''
        loss = self.lossFn1(self.stateMem,torch.tensor(envState,dtype=torch.float))
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()'''
    
    def actor_learn(self):
        # 每隔一定步数更新评估用网络的参数
        if self.learnStepCNT % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learnStepCNT+=1

        # 从经验池取样
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.expMem[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #Q*(s,a) = Q(s,a) + alpha*(r + gamma*max(Q(s',a')) - Q(s,a))
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.lossFn0(q_eval, q_target)

        self.optimizer0.zero_grad()
        loss.backward()
        self.optimizer0.step()

    def learn():
        pass

    def load(path):
        pass

    def save(path):
        pass