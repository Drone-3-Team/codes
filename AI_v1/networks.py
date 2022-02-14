from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hp
import matplotlib.pyplot as plt

# 深度预测网络.暂时不使用
class DepthPredict():
    def __init__(self) -> None:
        pass
    def forward(x):
        return x

# 根据深度图决策的行动网络
class Actor_Net(nn.Module):
    def __init__(self,args):
        super(Actor_Net, self).__init__()

        self.num_actions,self.num_states,self.env_a_shape,_ = args

        self.cov = nn.Sequential(
            nn.Conv3d(1,5,kernel_size=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=5),
            nn.Conv3d(5,1,kernel_size=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(2500, 50),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50,self.num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50,1),
        )

    def forward(self,x):
        x = self.cov(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        a = self.advantage(x)
        v = self.value(x)
        q = v.expand_as(a)+a-a.mean()
        return q

class DuelingDQN():

    def __init__(self,args) -> None:
        super(DuelingDQN, self).__init__()
        self.num_actions,self.num_states,self.env_a_shape,self.device = args
        self.eval_net, self.target_net = Actor_Net(args), Actor_Net(args)
        self.visual_net = DepthPredict()

        self.EpisodeCnt = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=hp.LR)
        self.lossFn = nn.MSELoss()

    def getDepth(self,Img):
        return DepthPredict.forward(Img)

    def predict(self,input):
        if np.random.randn() <= hp.EPISILO:# greedy
            action_value = self.eval_net.forward(input)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else: # random
            action = np.random.randint(0,self.num_actions)
            action = action if self.env_a_shape ==0 else action.reshape(self.env_a_shape)
        return action

    def visiual_learn(self,envState):
        pass
    
    def actor_learn(self,replay):
        # 每隔一定步数更新评估用网络的参数
        self.EpisodeCnt+=1
        if self.EpisodeCnt % hp.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            
        #Q*(s,a) = Q(s,a) + alpha*(r + gamma*max(Q(s',a')) - Q(s,a)))
        batch_state,batch_next_state,batch_reward,batch_action = replay
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        
        q_target = batch_reward + hp.GAMMA * q_next.max(1)[0].view(hp.BATCH_SIZE, 1)
        loss = self.lossFn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 保存已训练模型的参数
    def save(self,path):
        torch.save(self.eval_net, path)
    