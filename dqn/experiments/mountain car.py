from argparse import Action
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
MAX_ROUND = 233


env = gym.make("MountainCar-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n

env.reset()
action = np.random.randint(0,3)
state0 = env.step(action=action)
state = torch.unsqueeze(torch.FloatTensor(state0[0]), 0)
state_graph = env.render(mode='rgb_array')
print(state_graph.shape)


NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

def to_grayscle(rgb_array):
    grayscale_array = np.zeros([400,600],dtype=float)
    grayscale_array += rgb_array[:,:,0]
    grayscale_array += rgb_array[:,:,1]
    grayscale_array += rgb_array[:,:,2]
    grayscale_array = 1.0*grayscale_array/(255*3)
    return grayscale_array

class CNN_Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.vision = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(1, 15, kernel_size=2),
            nn.BatchNorm2d(15),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(15, 5, kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.fc0 = nn.Sequential(
            nn.Linear(5 * 38 * 58, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),  
        )
        self.fc1 = nn.Linear(128, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0,0.1)
        

    def forward(self,x):
        x = self.vision(x)
        ##print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
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
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.visual_net = CNN_Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #目标网络：
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1



        x_vision
        xreal

        #经验池：把智能体与环境交互得到的转移样本记下来，训练时从中随机取得batch
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]#从过往的轨道中随机选几个作为一个batch
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])#s
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))#a
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])#r
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #Q*(s,a) = Q(s,a) + alpha*(r + gamma*max(Q(s',a')) - Q(s,a))
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss0 = self.loss_func(q_eval, q_target)#网络的loss是均方误差
        loss1 = self.loss_func(x,)

        self.optimizer.zero_grad()
        loss0.backward()

        self.optimizer.step()


def reward_func(env, x, v,x_max,round_count):
    #这里奖励函数试图先让机器能登顶，再在登顶后最少化需要的操作步骤
    if x_max>=0.5:
        return x_max-0.25 - (round_count)/(MAX_ROUND*2.0)
    else:
        return x_max-0.25

def main():
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
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, v = next_state
            if x>max_x:
                max_x = x
            
            reward = reward_func(env, x, v, max_x,round_count)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn(x)
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done or round_count>MAX_ROUND:
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        print(max_x)

if __name__ == '__main__':
    main()