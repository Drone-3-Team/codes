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
        self.convVisualize = False
        plt.ion()

        self.conv = nn.Sequential(
            nn.Conv3d(1,10,kernel_size=(1,3,3),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=10),
            nn.Conv3d(10,10,kernel_size=(1,3,3),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=10),
            nn.Conv3d(10,1,kernel_size=(1,3,3),stride=(1,1,1))
        )

        self.fc = nn.Sequential(
            nn.Linear(1936, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,50),
            nn.ReLU(),
            nn.Linear(50,self.num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50,1),
        )

    def forward(self,x):
        x = self.conv(x)
        if self.convVisualize:
            plt.imshow((x.detach().cpu().numpy())[0,0,0])
            plt.pause(0.5)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        a = self.advantage(x)
        v = self.value(x)
        q = v.expand_as(a) + a-a.mean()
        return q

class DuelingDQN():

    def __init__(self,args) -> None:
        super(DuelingDQN, self).__init__()
        self.num_actions,self.num_states,self.env_a_shape,self.device = args
        self.eval_net, self.target_net = Actor_Net(args).to(self.device), Actor_Net(args).to(self.device)
        self.visual_net = DepthPredict()

        self.EpisodeCnt = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=hp.LR)
        self.lossFn = nn.MSELoss().to(self.device)

    def getDepth(self,Img):
        return DepthPredict.forward(Img)

    def predict(self,input):
        eps = 1-(self.EpisodeCnt+1/hp.EPISODES)+0.01
        if np.random.rand() <= eps:# greedy
            action_value = self.eval_net.forward(input)
            action = torch.max(action_value.cpu(), 1)[1].data.numpy()
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
    