# 一点点torch

torch提供的核心功能还是分为三大类：

    数据处理、模型构建、训练（自动调参）

## 数据处理

pytorch 有两种处理理数据集的工具

    torch.utils.data.DataLoader         给数据装在到一个可以自动迭代、采样的载体里
    torch.utils.data.Dataset            预制的数据集

使用DataLoader加载训练数据集

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    batch_size控制每一轮迭代中样本的个数

## 模型构建

### 继承nn.Module类来创建模型

实现 __init__(self)：用于初始化，网络的结构在这里定义；

实现 forward(self,x)：定义了网络前项传递的计算过程，调用在__init__中定义的层完成计算

### 模型编译、前向计算

使用一句

    model = NeuralNetwork().to(device)
    device可以是'cpu'、'cuda'

来将模型编译并移动到对应设备内存中

值得注意的是，torch的方法中往往含有device，例如torch.rand(shape,device = )。这凸显了torch的运算在不同设备上进行，例如使用cuda时rand运算应该在显卡上进行

tips:在调试时可以使用cpu，这样初始化所用的时间较短
6
此时可以向model中输入合适形状的向量并得到运算结果

### 层的分类

|名称|用途|参数|
|---|---|---|
|nn.Flatten|用于将高位结构展成一维（往往是图片）|N/A|
|nn.Linear|激活函数是线性的一层神经元|in_features:输入张量形状；out_features:输出张量形状|
|nn.ReLU|激活函数relu的一层神经元|初始化同上|
|nn.Sequential|按顺序堆叠的数层计算结构，它可以用于封装一些复杂的层|nn.Sequential(layer1,layer2,...)|
|nn.Softmax|一层Softmax神经元。其主要用途是归一化|dim = i对应张量第i维将被归一化|

## 自动调参
