import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(s):
    return s * (1 - s)

if __name__ == '__main__':
    # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
    v = np.asarray([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35]
    ])
    b1 = np.asarray([0.35])
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权    重系数w的形状为:[3,2]
    w = np.asarray([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ])
    b2 = np.asarray([0.65])

    # 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
    x = np.asarray([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = np.asarray([
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99]  # 人为任意给定的
    ])

    # 第一个隐藏的操作输出
    net_h = np.dot(x, v) + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
    out_h = sigmoid(net_h)
    # 输出层的操作输出
    net_o = np.dot(out_h, w) + b2  # [N,2] N表示样本数目，3表示每个样本有2个特征/2个输出
    out_o = sigmoid(net_o)
    loss = 0.5 * np.sum(np.power((out_o - d), 2))
    print(loss)
    print(net_h)
    print(out_h)
    print(net_o)
    print(out_o)
    print(x)
    print("=" * 50)

    # TODO: 基于矩阵的反向传播 --> 基于Numpy实现全连接神经网络

    # 反向传播
    delta_output = out_o - d
    grad_w = np.dot(out_h, delta_output)
    grad_b2 = np.sum(delta_output, axis=0)
    delta_hidden = np.dot(delta_output, w.T) * sigmoid_derivative(out_h)
    grad_v = np.dot(x.T, delta_hidden)
    grad_b1 = np.sum(delta_hidden,axis=0)

    # 打印梯度
    print("Gradient for w: ", grad_w)
    print("Gradient for B2: ", grad_b2)
    print("Gradient for v: ", grad_v)
    print("Gradient for B1: ", grad_b1)



