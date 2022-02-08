from myDQN import DQNnetwork
def run_this(TrainTime,StartLearnStep,DQN):
    step = 0
    for episode in range(TrainTime):
        #初始化环境
        observation = env.reset()
        while True:
            #刷新环境
            env.render()
            action = DQN.ChooseAction(observation)
            #根据环境，输入动作，获得奖励
            observation2,reward,done = env.step(action)
            DQN.StoreTransition(observation,action,reward,observation2)
            if(step > StartLearnStep)and (step%10 == 0):
                DQNnetwork.learn()
            #交换环境
            observation = observation2
            #游戏结束
            if done:
                break
        step = step+1
        print("训练结束一次")
        env.destory()#销毁环境

if __name__=='__main__':
    env = MakeEnvironment()
    DQN = DQNnetwork(
        env.allActions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replaceTargetIter=200,
        memorySize=2000
    )
    run_this(TrainTime=15000,StartLearnStep=500,DQN=DQN)
