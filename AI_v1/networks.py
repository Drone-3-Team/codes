from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hp

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
        self.num_actions,self.num_states,self.env_a_shape = args
        self.covLayer1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=5,kernel_size=(2,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=5),
        )

        self.covLayer2 = nn.Sequential(
            nn.Conv3d(in_channels=5,out_channels=5,kernel_size=(2,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(num_features=5),
        )
        self.fc1 = nn.Linear(37500, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,self.num_actions)
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

class DQN():

    def __init__(self,args) -> None:
        super(DQN, self).__init__()
        self.num_actions,self.num_states,self.env_a_shape = args
        self.eval_net, self.target_net = Actor_Net(args), Actor_Net(args)
        self.visual_net = DepthPredict()

        self.learnStepCNT = 0
        self.memCnt = 0
        self.stateMem = []
        self.nexStateMem = []
        for i in range(0,hp.BATCH_SIZE):
            self.stateMem.append(torch.Tensor(size = (1,1,hp.STACK_HEIGHT,hp.IMG_W,hp.IMG_H)))
            self.nexStateMem.append(torch.Tensor(size = (1,1,hp.STACK_HEIGHT,hp.IMG_W,hp.IMG_H)))
        self.expMem = np.zeros((hp.MEMORY_CAPACITY, 2))

        self.optimizer0 = torch.optim.Adam(self.eval_net.parameters(), lr=hp.LR)
        self.lossFn0 = nn.MSELoss()

    def getDepth(self,Img):
        return DepthPredict.forward(Img)

    def predict(self,state):
        if np.random.randn() <= hp.EPISILO:# greedy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else: # random
            action = np.random.randint(0,self.num_actions)
            action = action if self.env_a_shape ==0 else action.reshape(self.env_a_shape)
        return action

    def storeTransation(self,state,action,reward,nextState):
        index = self.memCnt % hp.MEMORY_CAPACITY        
        self.expMem[index,0] = action
        self.expMem[index,1] = reward
        self.stateMem[index] = state
        self.nexStateMem[index] = nextState
        self.memCnt += 1

    def visiual_learn(self,envState):
        '''
        loss = self.lossFn1(self.stateMem,torch.tensor(envState,dtype=torch.float))
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()'''
        pass
    
    def actor_learn(self):
        # 每隔一定步数更新评估用网络的参数
        self.learnStepCNT+=1
        if self.learnStepCNT % hp.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('.')
        else:
            return

        #Q*(s,a) = Q(s,a) + alpha*(r + gamma*max(Q(s',a')) - Q(s,a)))
        sample_index = np.random.choice(hp.MEMORY_CAPACITY, hp.BATCH_SIZE).astype(int).tolist()
        batch_ExpMem = self.expMem[sample_index]

        batch_action = torch.LongTensor(batch_ExpMem[:, 0].astype(int))
        batch_reward = torch.FloatTensor(batch_ExpMem[:, 1])
        eval_actions = torch.Tensor(size = (hp.BATCH_SIZE,self.num_actions))
        q_next = torch.Tensor(size = (hp.BATCH_SIZE,self.num_actions))
        for i in range(0,hp.BATCH_SIZE):
            eval_actions[sample_index[i]] = self.eval_net(self.stateMem[sample_index[i]])
            q_next[sample_index[i]] = self.target_net(self.nexStateMem[sample_index[i]])
        
        q_eval = eval_actions[:,batch_action]
        
        q_target = batch_reward + hp.GAMMA * q_next.max(1)[0].view(hp.BATCH_SIZE, 1)
        loss = self.lossFn0(q_eval, q_target)

        self.optimizer0.zero_grad()
        loss.backward()
        self.optimizer0.step()

    # 保存已训练模型的参数
    def save(self,path):
        torch.save(self.eval_net, path)
    