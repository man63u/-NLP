## RNN
### 特点：记忆功能

![RNN Image](https://github.com/user-attachments/assets/9f67d2d3-d7f9-465a-808a-4f3a0b2da26c)

### RNN结构
- \( X_t \)：时间 \( t \) 处的输入
- \( S_t \)：时间 \( t \) 处的记忆  
  \[ S_t = f(UX_t + WS_{t-1}) \]
  其中 \( f \) 可以是非线性转换函数（如 \( \tanh \)）
- \( O_t \)：时间 \( t \) 处的输出（如sigmoid/softmax输出的属于每个候选词的概率）
- \( h_t \)：时间 \( t \) 的隐藏状态
- \( U \)：输入层到隐藏层间的权重
- \( W \)：隐藏层到隐藏层的权重，负责调度记忆
- \( V \)：隐藏层到输出层，做为最后 一次抽象

### RNN正向传播过程

![RNN Forward Propagation](https://github.com/user-attachments/assets/87f8a953-16ec-4962-9c89-c5f703ea09e4)
公式：
t=a 
$$ S_a=UX_a+WS_{a-1}
h_a=f(UX_a+WS_{a-1})
O_a=g(Vh_a) $$ 
### RNN反向传播
用链式法则，求V梯度（不存在和之前的状态依赖），求导->求和


 
