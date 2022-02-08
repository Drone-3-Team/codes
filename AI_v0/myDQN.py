import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Sequential, Flatten,Linear


class DQNnetwork(nn.Module):
    def __init__(
            self,
            allAction,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replaceTargetIter=300,
            memorySize=500,
            batchSize=32,
            e_greedy_increment=None,
    ):
        self.allAction = allAction
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replaceTargetIter = replaceTargetIter  # 更换 target_net 的步数
        self.memorySize = memorySize  # 记忆上限
        self.batchSize = batchSize  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learnStepCounter = 0

        # 初始化全 0 记忆 [observation, action, reward, observation2]
        self.memory = np.zeros((self.memorySize, n_features * 2 + 2))  # 和视频中不同, 因为 pandas 运算比较慢, 这里改为直接用 numpy

        # 创建 [target_net, evaluate_net]
        self.eval_net, self.target_net = Qfunction(),Qfunction()

    def store_transition(self, observation, action, reward, observation2):
        # 记录一条 [observation, action, reward, observation2)] 记录
        transition = np.hstack((observation, [action, reward], observation2))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memoryCounter % self.memorySize
        self.memory[index, :] = transition  # 替换一条memory
        self.memoryCounter += 1

    def ChooseAction(self, observation):
        #抽取随机数，如果小于临界值则根据Q函数选择行为，否则随机选择行为，保持agent的好奇心
        if np.random.uniform() < self.epsilon:
            actionValue = self.eval_net(observation)
            action = np.argmax(actionValue)
        else:
            action = np.random.randint(0, self.allAction)  # 随机选择
        return action

    def learn(self):
        #每隔几步更新参数
        if self.learnStepCounter % self.replaceTargetIter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learnStepCounter += 1

        # 从 memory 中随机抽取 batch_size 这么多记忆
        sampleIndex = np.random.choice(self.memorySize, size=self.batchSize)
        batchMemory = self.memory[sampleIndex, :]

        #计算两个网络对当前状态产生的q值以及误差
        q_next, q_eval = self.target_net(batchMemory[:, -self.n_features:]),self.eval_net(batchMemory[:, self. n_features])
        q_target = q_eval.copy()
        batchIndex = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batchMemory[:, self.n_features].astype(int)
        reward = batchMemory[:, self.n_features + 1]
        q_target[batchIndex, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #计算误差，改变参数
        loss = nn.MSELoss(q_target, q_eval)
        optimzer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learnStepCounter += 1

class Qfunction(nn.Module):
    def __init__(self):
        super(DQNnetwork,self).__init__()
        #Q network
        self.model1 = Sequential(
            Conv2d(4,32,5,padding=2,stride=8),# 输入四张深度图像，通道数为4,输出通道数为32，卷积核用5*5，步长8，计算得padding=2*2
            Conv2d(32,64,3,padding=1,stride=2),#输入通道数为32,输出通道数为64，卷积核用3*3，步长2，计算得padding=1*1
            Conv2d(64, 64, 3, padding=1, stride=1), #输入通道数为64,输出通道数为64，卷积核用3*3，步长1，计算得padding=1*1
            Flatten(),
            Linear(5120,512),
            Linear(512,4)#控制前后左右四个动作表示四个动作的Q值
        )

    def forward(self,x):
        x = self.model1(x)
        return x
