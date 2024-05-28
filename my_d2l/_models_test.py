import torch
from torch import nn

import data_preprocess
import net_model
import loss_model
import updater_model
import train_model


# 线性回归
def linear_scratch():
    """线性回归模型——从零开始实现"""
    features, labels = data_preprocess.synthetic_data(torch.tensor([2.9, 5.8]), 8.2, 1000)
    num_epochs = 3
    batch_size = 10
    lr = 0.03

    net = net_model.linreg
    W = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss = loss_model.squared_loss
    updater = updater_model.sgd

    train_model.linear_scratch_train(num_epochs, batch_size, lr, features, labels, net, [W, b], loss, updater)


def linear_concise():
    """线性回归模型——简洁实现"""
    features, labels = data_preprocess.synthetic_data(torch.tensor([2.9, 5.8]), 8.2, 1000)
    num_epochs = 3
    batch_size = 10
    lr = 0.03

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr)

    train_model.linear_concise_train(num_epochs, batch_size, features, labels, net, loss, trainer)


# 线性分类
def softmax_scratch():
    """线性分类模型——从零开始实现"""
    num_epochs = 5
    batch_size = 256
    lr = 0.1
    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)

    net = net_model.linClassify
    W = torch.normal(0, 0.01, size=(28 * 28, 10), requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    loss = loss_model.cross_entropy
    updater = updater_model.sgd

    train_model.softmax_scratch_train(num_epochs, batch_size, lr, train_iter, test_iter, net, [W, b], loss, updater)


def softmax_concise():
    """线性分类模型——简洁实现"""
    num_epochs = 5
    batch_size = 256
    lr = 0.1
    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))  # softmax集成在损失函数CrossEntropyLoss中，因此不需要显式表示
    net.apply(net_model.init_weights)
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)

    train_model.softmax_concise_train(num_epochs, train_iter, test_iter, net, loss, updater)


# 多层感知机
def MLP_scratch():
    num_epochs = 10
    batch_size = 256
    lr = 0.1
    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
    b2 = torch.zeros(num_outputs, requires_grad=True)

    net = net_model.MLP
    params = [W1, b1, W2, b2]
    loss = loss_model.cross_entropy
    updater = updater_model.sgd
    train_model.softmax_scratch_train(num_epochs, batch_size, lr, train_iter, test_iter, net, params, loss, updater)



def MLP_concise():
    num_epochs = 5
    batch_size = 256
    lr = 0.1
    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    net.apply(net_model.init_weights)
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(net.parameters(), lr)

    train_model.softmax_concise_train(num_epochs, train_iter, test_iter, net, loss, updater)




if __name__ == "__main__":
    # print('从零开始实现线性回归模型：')
    # linear_scratch()
    # print('简洁实现线性回归模型：')
    # linear_concise()
    # print('从零开始实现softmax分类模型：')
    # softmax_scratch()
    # print('简洁实现softmax分类模型：')
    # softmax_concise()
    print('从零开始实现多层感知机模型：')
    MLP_scratch()
    # print('简洁实现多层感知机模型：')
    # MLP_concise()
