import torch
import data_preprocess
import evaluate
import d2l_tools
# from d2l import torch as d2l


# 线性回归
def linear_scratch_train(num_epochs, batch_size, lr, features, labels, net, params, loss, updater):
    for epoch in range(num_epochs):
        for X, y in data_preprocess.data_iter_by_hand(batch_size, features, labels):
            l = loss(net(X, params[0], params[1]), y)         # 正向传播
            l.sum().backward()                  # 反向传播
            updater(params, lr, batch_size)     # 更新参数
        with torch.no_grad():
            train_l = loss(net(features, params[0], params[1]), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


def linear_concise_train(num_epochs, batch_size, features, labels, net, loss, updater):
    for epoch in range(num_epochs):
        for X, y in data_preprocess.load_array((features, labels), batch_size):
            l = loss(net(X), y)
            updater.zero_grad()
            l.backward()
            updater.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


# 线性分类
def softmax_scratch_train(num_epochs, batch_size, lr, train_iter, test_iter, net, params, loss, updater):
    """softmax训练"""
    for epoch in range(num_epochs):
        metric = d2l_tools.Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
        for X, y in train_iter:
            y_hat = net(X, params[0], params[1])    # 正向传播
            l = loss(y_hat, y)      # 计算损失
            l.sum().backward()      # 反向传播
            updater(params, lr, batch_size)   # 更新参数
            metric.add(float(l.sum()), evaluate.accuracy(y_hat, y), y.numel())
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, acc {train_metrics[1]:.5f}')
    test_acc = evaluate.evaluate_accuracy_scratch(net, test_iter, params[0], params[1])
    print(f'完成训练，在测试集上 acc {test_acc:.5f} \n')


def softmax_concise_train(num_epochs, train_iter, test_iter, net, loss, updater):
    """softmax训练"""
    # animator = tools.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l_tools.Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
        for X, y in train_iter:
            y_hat = net(X)  # 计算梯度并更新参数
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            metric.add(float(l.sum()), evaluate.accuracy(y_hat, y), y.numel())
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, acc {train_metrics[1]:.5f}')
        # test_acc = tools.evaluate_accuracy(net, test_iter, W, b)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    test_acc = evaluate.evaluate_accuracy_concise(net, test_iter)
    print(f'完成训练，在测试集上 acc {test_acc:.5f}')