import torch
from torch import nn


def linreg(X, W, b):
    """线性回归模型——1层——全连接层"""
    return torch.matmul(X, W) + b


def softmax(X):
    """softmax模型——1层——softmax层"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 广播机制


def linClassify(X, W, b):
    """线性分类模型——2层——全连接层+softmax层"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def init_weights(m):
    """自动pytorch网络参数初始化"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)