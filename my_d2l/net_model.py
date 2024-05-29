import torch
from torch import nn

# 操作子和激活函数
def softmax(X):
    """softmax操作子"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 广播机制


def relu(X):
    """relu激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1 # 确保丢弃概率合法
    # 所有元素都丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 所有元素都保留
    if dropout == 0:
        return X
    mask = torch.rand(X.shape) > dropout
    return mask * X / (1.0 - dropout)

# 网络结构
def linreg(X, params):
    """线性回归——1层——输出层"""
    return torch.matmul(X, params[0]) + params[1]


def linClassify(X, params):
    """线性分类——1层——输出层"""
    return softmax(torch.matmul(X.reshape((-1, params[0].shape[0])), params[0]) + params[1])


def MLP(X, params):
    """多层感知机——2层——隐藏层+输出层"""
    X = X.reshape((-1, 784))
    H = relu(X @ params[0] + params[1])
    return softmax(H @ params[2] + params[3])


# 网络参数初始化
def init_weights(m):
    """自动pytorch网络参数初始化"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
