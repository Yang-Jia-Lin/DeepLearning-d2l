import torch
from torch import nn
from torch.nn import functional as F

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

# 从零开始实现网络结构
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


# 简洁实现网络结构
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # module是Module子类的一个实例, 把它保存在'Module'类的成员中
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module # 变量_modules中。_module的类型是OrderedDict

    def forward(self, X):
        # OrderedDict保证按照成员添加顺序遍历
        for block in self._modules.values():
            X = block(X)
        return X


class MLP_nn(nn.Module):
    # 参数声明
    def __init__(self):
        super().__init__() # 调用MLP的父类Module的构造函数来初始化。这样也可以指定其他函数参数（模型参数params）
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 前向传播
    def forward(self, X):
        return self.out(F.relu(self.hidden(X))) # ReLU的函数版本，其在nn.functional模块中定义。


# 网络参数初始化
def init_weights(m):
    """自动pytorch网络参数初始化"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
