import torch
import data_preprocess
import train_model
import net_model
import loss_model
import updater_model
from torch import nn


def linear_scratch():
    """线性回归模型——从零开始实现"""
    features, labels = data_preprocess.synthetic_data(torch.tensor([2.9, 5.8]), 8.2, 1000)
    num_epochs = 3
    batch_size = 10
    lr = 0.03

    net = net_model.linreg
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss = loss_model.squared_loss
    updater = updater_model.sgd

    train_model.linear_scratch_train(num_epochs, batch_size, lr, loss, updater, net, features, labels, w, b)


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

    train_model.linear_concise_train(num_epochs, batch_size, loss, trainer, net, features, labels)


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

    train_model.softmax_scratch_train(net, train_iter, test_iter, loss, num_epochs, updater, W, b, lr)


def softmax_concise():
    """线性分类模型——简洁实现"""
    num_epochs = 5
    batch_size = 256
    lr = 0.1
    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(net_model.init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr)

    train_model.softmax_concise_train(net, train_iter, test_iter, loss, num_epochs, trainer)