from d2l_tools import Accumulator
from d2l_tools import argmax, astype, reduce_sum


def accuracy(y_hat, y):
    """一个批量中预测正确数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果预测是二维的且有多个输出（多类别的预测结果）
        y_hat = argmax(y_hat, axis=1)  # 使用 argmax 取最值
    cmp = astype(y_hat, y.dtype) == y  # 比较预测和真实结果（得到布尔数组）
    return float(reduce_sum(astype(cmp, y.dtype)))  # 统计正确的数量并转化为float输出


def evaluate_accuracy_scratch(net, data_iter, W, b):
    """数据集上的预测准确率——手动实现"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X, W, b), y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_concise(net, data_iter):
    """数据集上的预测准确率——自动实现"""
    net.eval()  # 不累计梯度
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
