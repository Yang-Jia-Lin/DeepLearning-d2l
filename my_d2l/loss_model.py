import torch


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat, y):
    """交叉损失熵"""
    # y_hat 形状：[N,C]的矩阵，表示对于每一个样本分类的概率预测
    # y 形状：[N]的向量，表示每一个样本的实际类别编码
    return - torch.log(y_hat[range(len(y_hat)), y])
    # range(N)是行索引（有N个元素），y是列索引（也有N个元素）


# 权重衰减（惩罚项）
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2