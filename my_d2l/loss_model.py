import torch


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat, y):
    """交叉损失熵"""
    return - torch.log(y_hat[range(len(y_hat)), y])