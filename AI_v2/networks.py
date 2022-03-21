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
            nn.Conv3d(1,5,kernel_size=(1,3,3),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=5),
            nn.Conv3d(5,5,kernel_size=(1,3,3),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=5),
            nn.Conv3d(5,1,kernel_size=(1,3,3),stride=(1,1,1))
        )

        self.fc = nn.Sequential(
            nn.Linear(1936, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.pos = nn.Sequential(
            nn.Linear(7,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,50),
            nn.ReLU(),
            nn.Linear(50,self.num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.Linear(50,1),
        )

    def forward(self,x,y):
        x = self.conv(x)
        if self.convVisualize:
            plt.imshow((x.detach().cpu().numpy())[0,0,0])
            plt.pause(0.5)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        y = self.pos(y)
        m = torch.cat([x,y],dim=1) #将抽象后的视觉信息与位置信息拼接
        #print(m.size())
        a = self.advantage(m)
        v = self.value(m)
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

    def getDepth(self,Img): #此处现已废弃。原本是用作图像转深度图
        return DepthPredict.forward(Img)

    def predict(self,input):
        # 使得随机程度随着训练的进行下降
        eps = (self.EpisodeCnt/hp.EPISODES)+0.3
        Img,pos = input
        if np.random.rand() <= eps:# 选择贪心
            action_value = self.eval_net.forward(Img,pos)
            action = torch.max(action_value.cpu(), 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else: # 选择随机策略
            action = np.random.randint(0,self.num_actions)
            action = action if self.env_a_shape ==0 else action.reshape(self.env_a_shape)
        return action

    def visiual_learn(self,envState): #已废弃，训练图像转深度图的方法
        pass
    
    def actor_learn(self,replay):
        # 每隔一定步数更新评估用网络的参数
        if self.EpisodeCnt % hp.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            
        #Q*(s,a) = Q(s,a) + alpha*(r + gamma*max(Q(s',a')) - Q(s,a)))
        batch_state,batch_pos,batch_next_state,batch_nex_pos,batch_reward,batch_action = replay

        q_eval = self.eval_net(batch_state,batch_pos).gather(1, batch_action)
        q_next = self.target_net(batch_next_state,batch_nex_pos).detach()
        q_target = batch_reward + hp.GAMMA * q_next.max(1)[0].view(hp.BATCH_SIZE, 1)

        loss = self.lossFn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 保存已训练模型的参数
    def save(self,path):
        torch.save(self.eval_net, path)
    