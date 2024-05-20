import torch
import data_preprocess
import tools
# from d2l import torch as d2l


# 线性回归训练
def linear_concise_train(num_epochs, batch_size, loss, trainer, net, features, labels):
    for epoch in range(num_epochs):
        for X, y in data_preprocess.load_array((features, labels), batch_size):
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
def linear_scratch_train(num_epochs, batch_size, lr, loss, trainer, net, features, labels, w, b):
    for epoch in range(num_epochs):
        for X, y in data_preprocess.data_iter_by_hand(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 正向传播
            l.sum().backward()  # 反向传播
            trainer([w, b], lr, batch_size)  # 更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# 线性分类训练
def softmax_scratch_train(net, train_iter, test_iter, loss, num_epochs, updater, W, b, lr):  #@save
    """softmax训练"""
    for epoch in range(num_epochs):
        metric = tools.Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
        for X, y in train_iter:
            y_hat = net(W, b, X)  # 计算梯度并更新参数
            l = loss(y_hat, y)
            l.sum().backward()
            updater(W, b, lr, X.shape[0])
            metric.add(float(l.sum()), tools.accuracy(y_hat, y), y.numel())
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, acc {train_metrics[1]:.5f}')
    test_acc = tools.evaluate_accuracy_scratch(net, test_iter, W, b)
    print(f'完成训练，在测试集上 acc {test_acc:.5f} \n')
def softmax_concise_train(net, train_iter, test_iter, loss, num_epochs, updater):
    """softmax训练"""
    # animator = tools.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        metric = tools.Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
        for X, y in train_iter:
            y_hat = net(X)  # 计算梯度并更新参数
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            metric.add(float(l.sum()), tools.accuracy(y_hat, y), y.numel())
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, acc {train_metrics[1]:.5f}')
        # test_acc = tools.evaluate_accuracy(net, test_iter, W, b)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    test_acc = tools.evaluate_accuracy_concise(net, test_iter)
    print(f'完成训练，在测试集上 acc {test_acc:.5f}')