import torch
import data_preprocess
import train
from torch import nn

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def linear_scratch(features, labels):
    """线性模型的从零开始实现"""
    num_epochs = 3
    batch_size = 10
    lr = 0.03

    net = linreg
    loss = squared_loss
    trainer = sgd
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    train.linear_scratch_train(num_epochs, batch_size, lr, loss, trainer, net, features, labels, w, b)


def linear_concise(features, labels):
    """线性模型的简洁实现"""
    num_epochs = 3
    batch_size = 10
    lr = 0.03

    net = nn.Sequential(nn.Linear(2, 1))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr)
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    train.linear_concise_train(num_epochs, batch_size, loss, trainer, net, features, labels)


if __name__ == "__main__":
    true_w = torch.tensor([2.9, 5.8])
    true_b = 8.2
    print('从零开始实现线性回归模型：')
    features, labels = data_preprocess.synthetic_data(true_w, true_b, 1000)
    linear_scratch(features, labels)
    print('简洁实现线性回归模型：')
    features, labels = data_preprocess.synthetic_data(true_w, true_b, 1000)
    linear_concise(features, labels)