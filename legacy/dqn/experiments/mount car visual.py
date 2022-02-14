import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# environment
env = gym.make("MountainCar-v0")
env = env.unwrapped

# hyper parameters
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

def stack_image(stack,newImage):
    newImage = torch.unsqueeze(newImage, dim = 0)    # 添加一个维度以使用cat
    stacked = torch.cat([stack[1:],newImage],dim = 0)   #拼接，去除stack[1,:,:]之前的元素
    return stacked

def addDimForCNN(imgStack):
    tensor5D = torch.zeros(1,1,STACK_HEIGHT,IMG_H,IMG_W)
    tensor5D[0,0] = imgStack
    return tensor5D

def getCNNInput(imgStack,new_RGBimg):
    newImg = img2tensor(new_RGBimg)
    newStack = stack_image(imgStack,newImg)
    return newStack,addDimForCNN(newStack)

class Vision_Net(nn.Module):
    def __init__(self) -> None:
        super(Vision_Net, self).__init__()
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

        self.fcLayer = nn.Sequential(
            nn.Linear(240,20),
            nn.ReLU(inplace=True),
            nn.Linear(20,2),
        )
    
    def forward(self,x):
        x = self.covLayer1(x)
        x = self.covLayer2(x)
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.fcLayer(x)
        return x

class Actor_Net(nn.Module):
    def __init__(self):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():

    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Actor_Net(), Actor_Net()
        self.visual_net = Vision_Net()

        self.learnStepCNT = 0
        self.memCnt = 0
        self.expMem = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.stateMem = torch.rand(2,dtype=float)

        self.optimizer0 = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.lossFn0 = nn.MSELoss()
        self.optimizer1 = torch.optim.Adam(self.visual_net.parameters(), lr=LR)
        self.lossFn1 = nn.L1Loss()

        self.imgStack = torch.ones([STACK_HEIGHT,IMG_H,IMG_W],dtype=torch.float)

    def getCarState(self,newRGBImg):
        self.imgStack, Vtensor = getCNNInput(self.imgStack,newRGBImg)
        self.stateMem = self.visual_net(Vtensor)[0]
        return self.stateMem

    def act(self,state):
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
        loss = self.lossFn1(self.stateMem,torch.tensor(envState,dtype=torch.float))
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()
    
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
        print(batch_state.size())
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.lossFn0(q_eval, q_target)

        self.optimizer0.zero_grad()
        loss.backward()
        self.optimizer0.step()

def rewardFn(xMax):
    return xMax

if __name__ == '__main__':
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        round_count = 0
        max_x = -10.0
        state = env.reset()
        ep_reward = 0
        while True:
            round_count += 1

            rgbImg = env.render('rgb_array')
            statep = dqn.getCarState(rgbImg)
            dqn.visiual_learn(state)
            #print(statep.detach().numpy()-state,statep.detach().numpy(),state)
            
            action = dqn.act(statep)
            #print(action)
            next_state, _ , done, info = env.step(action)
            x, v = next_state

            if x>max_x:
                max_x = x
            
            reward = rewardFn(max_x)

            dqn.storeTransation(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memCnt >= MEMORY_CAPACITY:
                dqn.actor_learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done or round_count>MAX_ROUND:
                break
            state = next_state

        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        print(max_x)