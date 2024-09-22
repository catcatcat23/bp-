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
    b1 = np.asarray([0.35, 0.35, 0.35])
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
    w = np.asarray([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ])
    b2 = np.asarray([0.65, 0.65])

    # 输入数据
    x = np.asarray([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = np.asarray([
        [0.01, 0.99],
        [0.01, 0.99],
        [0.01, 0.99],
        [0.01, 0.99],
        [0.01, 0.99]
    ])

    lr = 0.5  # 学习率

    for epoch in range(10000):
        # 前向传播
        net_h = np.dot(x, v) + b1  # 隐藏层输入 [N,3]
        out_h = sigmoid(net_h)  # 隐藏层输出 [N,3]
        net_o = np.dot(out_h, w) + b2  # 输出层输入 [N,2]
        out_o = sigmoid(net_o)  # 输出层输出 [N,2]

        # 计算损失
        loss = 0.5 * np.sum(np.power(out_o - d, 2))

        # 反向传播
        t = (out_o - d) * sigmoid_derivative(out_o)  # 输出层误差 [N,2]

        # 更新输出层的权重和偏置
        w -= lr * np.dot(out_h.T, t)  # 更新权重 [3,2]
        b2 -= lr * np.sum(t, axis=0)  # 更新偏置 [2]

        # 计算隐藏层误差
        delta_h = np.dot(t, w.T) * sigmoid_derivative(out_h)  # 隐藏层误差 [N,3]

        # 更新隐藏层的权重和偏置
        v -= lr * np.dot(x.T, delta_h)  # 更新权重 [2,3]
        b1 -= lr * np.sum(delta_h, axis=0)  # 更新偏置 [3]

        if epoch % 1000 == 0:
            print('Loss: {:.10f}'.format(loss))






    
