from d2l import torch as d2l
from tools import Accumulator


def accuracy(y_hat, y):
    """计算预测正确的个数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


def evaluate_accuracy_scratch(net, data_iter, W, b):
    """计算指定数据集上的模型精度——手动实现需要 W,b"""
    metric = Accumulator(2)  #正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(W, b, X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_concise(net, data_iter):
    """计算指定数据集上的模型精度——自动实现"""
    net.eval()  #不累计梯度
    metric = Accumulator(2)  #正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]