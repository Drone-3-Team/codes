# codes

## 仓库结构

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

## 最近改动：

1、团队的仓库folk了钻洞、gym的源码

2、上传了dqn的学习内容，包括一些笔记（关于dqn、numpy、pandas）以及一些gym中的测试代码

3,myDQN为对第三章介绍的网络结构的实现，q函数输出为前后左右四个动作的打分，结合main文件使用，还差结合无人机和环境训练部分的代码，图像输入结合深度图像文件中训练好的数据使用，不过还未实现，代码可能有bug，欢迎好兄弟们继续在我的基础上继续实现
