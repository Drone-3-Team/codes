# Dueling DQN

## 结构

Dueling DQN是普通DQN的一种改型。它将 价值 $Q$ 拆分为 价值 $V$ 与 优势 $A$ 的合

$$Q(S,A,\omega,\alpha,\beta) = A(S,A,\omega,\beta) + V(S,\omega,\alpha,\beta)$$

据此将DQN网络的结构修改：

![avator](/notes/notepic/Dueling-DQN.jpg)

其中变量含义：

|||
|--|--|
|$S$|当前状态|
|$A$|要采取的行动|
|$\omega$|Dueling-DQN网络公共部分的参数|
|$\alpha$|Dueling-DQN网络价值函数独有的参数|
|$\beta$|Dueling-DQN网络优势函数独有的参数|

为了在训练时正确辨识V、A对Q的影响，对A做了去中心化处理，公式变为：

$$Q(S,A,\omega,\alpha,\beta) = V(S,\omega,\alpha) + (A(S,A,\omega,\beta) - \frac{1}{\Alpha}\sum_{a' \in \Alpha} A(S,a',\omega,\beta))$$

这样一来确保了一个情况下对于所有行动，A的和为0，这样A值仅于当前采取的行动有关，V值与状态有关，二者的更新的原因便得以区分

## 优势

Dueling-DQN较之普通版本，主要优势在于它能更准确地估计Q值。

在应用中有时采取何种行动不关乎结果，有时相反。Dueling-DQN的A部分可以在一些“立即关乎结果”的事件上做出反应。

![avator](/notes/notepic//DuelingDQNsample.jfif)

在经过训练的Dueling-DQN网络上，当前方空旷时Q更多与“更为长远”的路线曲度有关，此时Q被V主导。当存在碰撞风险时，Q与“更为紧急”的碰撞有关，此时Q受到了A的显著影响

## 训练

