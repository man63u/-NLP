## 模型结构
自编码器的作用：
 - 数据降维
 - 数据去噪
 - 特征学习
   结构：
- 编码器
- 解码器
## 代码结构解析
- 构造函数（——init——）
  主要属性：
  - max_seq_length:解码过程中最大序列长度
  - eos_token_id:表示解码结束的特殊token id
- 向前过程(forward)
  状态初始化
  模式：
   - 训练模式（self_training=True):全序列输入
   - 推理模式（self_training=False)：逐时刻生成输出
## seq2seq2 attention
![image](https://github.com/user-attachments/assets/2b3705d8-3835-4794-a50d-38b430420f81)
