# codes

## 有用的链接

🔗<https://zhuanlan.zhihu.com/p/38056115>🔗torch存取模型以及使用预训练模型

🔗<https://zhuanlan.zhihu.com/p/31778008>🔗一些关于无人机避障方法的介绍，以及在dji pantom上的实现

🔗<https://github.com/microsoft/AirSim/blob/master/PythonClient/reinforcement_learning/dqn_drone.py>🔗
🔗<https://microsoft.github.io/AirSim/reinforcement_learning/#gym-wrapper>🔗
使用airsim构建gym风格的强化学习环境

🔗<https://github.com/DLR-RM/stable-baselines3>🔗stable-baselines3.内含很多现成的算法

## 仓库结构

    AI_v1：第一版模型
    |   所实现的功能是无人机钻洞，替换了网上那个项目中自带的模型
    └   使用airsim环境，模型采用了RGB转深度的网络以及一个决策网络，训练过程使用GYM管理

    airsim：一些配置airsim用到的文件
    └   包含json、vs project文件、unreal project文件

    dqn：深度强化学习的相关代码
    ├   data：pytorch的数据集路径
    |       自动下载的数据集，可无视
    |
    ├experiments：我做成的实验代码
    |       目前有爬山小车非视觉版、视觉版
    |
    ├shelperCodes：一些运行torch函数的例子
    └       备有较详细注释的简单样例，可能有助于理解pytorch特性

    notes：我的一点笔记，只有我自己能看懂

## 最近改动

3、myDQN为对第三章介绍的网络结构的实现，q函数输出为前后左右四个动作的打分，结合main文件使用，还差结合无人机和环境训练部分的代码，图像输入结合深度图像文件中训练好的数据使用，不过还未实现，代码可能有bug，欢迎好兄弟们继续在我的基础上继续实现

4、开工Airsim+Gym+dqn的钻洞试验了。成功后的模型稍加修改即可用在第一版上
