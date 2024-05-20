import torch
import data_preprocess
import train
from torch import nn


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
def net(W, b, X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
def updater(W, b, lr, batch_size):
    """小批量随机梯度下降，更新参数"""
    params = [W, b]
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def softmax_scratch():
    """线性模型的从零开始实现"""
    num_epochs = 5
    batch_size = 256
    lr = 0.1

    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)
    num_inputs = 28 * 28
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    net_scratch = net
    loss = cross_entropy
    trainer = updater
    train.softmax_scratch_train(net_scratch, train_iter, test_iter, loss, num_epochs, trainer, W, b, lr)


def softmax_concise():
    """线性模型的简洁实现"""
    num_epochs = 5
    batch_size = 256
    lr = 0.1

    train_iter, test_iter = data_preprocess.load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr)
    train.softmax_concise_train(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    print('从零开始实现softmax分类模型：')
    softmax_scratch()
    print('简洁实现softmax分类模型：')
    softmax_concise()