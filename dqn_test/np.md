# 一点numpy

以及一点点python

## array

### 创建/初始化

#### np.array()

参数:

    一个序列，初始化的数据
    dtype：数据类型（np.int,float,etc）

#### np.xxx()

|xxx|类型|
|---|---|
|zeros|全0|
|ones|全1|
|empty|接近0的数|
|arange(begin,end,step)|整数等差|
|linspace(begin,end,num)|浮点等差|
|np.random.randon((shape))

参数:
    (shape)
    dtype

#### np.reshape()

参数：
    (shape)

### 成员

array 中有以下成员变量

|名称|作用|
|----|----|
|ndim|维数|
|shape|矩阵形状|
|size|元素个数|
|dtype|数据类型|

array中有以下成员函数

## 运算

### 基础运算

注意这里的基础指运算针对每个元素，而非矩阵运算

|符号|运算|
|---|---|
|+-*/|加减乘除|
|a**b|a的b次幂|
|np.tri()|三角函数，弧度制|
|<,==（偏序/逻辑）|返回判断后的布尔值|

### 矩阵运算

|符号|运算|
|---|---|
|np.dot(a,b);a.dot(b)|矩阵乘|

### 统计性质

    np.sum(a,axis)
    np.min(a,axis);.max(a,axis)
    np.argmin();np.argmax() # 返回索引（下标）值
    np.mean()
    np.median()     #   中位数
    np.cumsum()     #   前缀和
    np.diff()       #   差分
    np.nonsero()    #   非0布尔值

axis是操作对应的维度。对于二维的矩阵，0是在列中操作，1是行

### 形状操作

合并、分割、转置之类的

#### 合并

|符号|运算|
|---|---|
|.vstack|行堆叠，序列合并后还会升维,可以合并多个|
|.hstack|列粘连，同上|
|.newaxis|A(np.newaxis,:) 在对应位置添加一个新维度|
|.concatenate((),axis = )|指定维度并合并|

#### 分割

|符号|运算|
|---|---|
|.split(A,divnum,axis)|沿axis等分为divnum块，返回一个元素是array的矩阵|
|.array_split(A,divnum,axis)|允许不等分|
|.vsplit/.hsplit(A,num)|分割为num块|

## python 特性

>默认赋值只赋指针值 可以用 A is B 判断B是否是A的指针

 要拷贝内容，使用B = A.copy()

>python中:是一种切片操作 左界:右界（闭区间）。当缺省界限时则是上/下界
同时右界可以用-i等元素表示倒数第i个元素。

 当下标不合法时（左界>右界），不会报错，所得的序列为空
