import copy
import torch
import matplotlib.pyplot as plt
import scripts.airsim_env as env
from networks import DQN
import hp

class Agent():
    '''
    用于管理DQN训练
    TODO: 训练多个智能体,从中选优
    '''
    def __init__(self,trainEnv) -> None:
        self.dqn = DQN()
        self.trainEnv = trainEnv
    
    def learn(self):
        reward_list = []
        plt.ion()
        fig, ax = plt.subplots()
        for i in range(hp.EPISODES):
            round_count = 0
            max_x = -10.0
            state = self.trainEnv.reset()
            ep_reward = 0
            while True:
                round_count += 1

                Img = self.trainEnv.get_obs()
                statep = self.dqn.getDepth(Img)
                self.dqn.visiual_learn(state)
                
                action = self.dqn.predict(statep)
                #print(action)
                next_state, _ , done, info = self.trainEnv.step(action)
                self.trainEnv.do_action(action)
                
                reward = self.trainEnv.compute_reward()

                self.dqn.storeTransation(state, action, reward, next_state)
                ep_reward += reward

                if self.dqn.memCnt >= hp.MEMORY_CAPACITY:
                    self.dqn.actor_learn()
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                if done or round_count>hp.MAX_ROUND:
                    break
                state = next_state

            r = copy.copy(reward)
            reward_list.append(r)
            ax.set_xlim(0,300)
            ax.plot(reward_list, 'g-', label='total_loss')
            plt.pause(0.001)
            print(max_x)

    def predict(self,obs):
        return self.dqn.predict(obs)

    def save(self):
        self.dqn.save(hp.path+'text_model.pth')

    def load(self,path):
        self.dqn.eval_net = self.dqn.target_net = torch.load(path)
