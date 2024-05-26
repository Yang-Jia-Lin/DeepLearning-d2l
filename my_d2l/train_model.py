import torch
import data_preprocess
import evaluate
import d2l_tools


# 线性回归
def linear_scratch_train(num_epochs, batch_size, lr, features, labels, net, params, loss, updater):
    """线性回归模型训练实现——手动"""
    for epoch in range(num_epochs):
        # 模型训练
        for X, y in data_preprocess.data_iter_by_hand(batch_size, features, labels):
            # 1. 正向传播+计算损失
            l = loss(net(X, params[0], params[1]), y)
            # 2. 反向传播
            l.sum().backward()
            # 3. 更新参数
            updater(params, lr, batch_size)
        # 模型评估
        with torch.no_grad():
            train_l = loss(net(features, params[0], params[1]), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print('完成训练\n')


def linear_concise_train(num_epochs, batch_size, features, labels, net, loss, updater):
    """线性回归模型训练实现——自动"""
    for epoch in range(num_epochs):
        # 模型训练
        for X, y in data_preprocess.load_array((features, labels), batch_size):
            # 1.正向传播+计算损失
            l = loss(net(X), y)
            # 2.反向传播（先将梯度清零）
            updater.zero_grad()
            l.backward()
            # 3.更新参数
            updater.step()
        # 模型评估
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    print('完成训练\n')


# 线性分类
def softmax_scratch_train(num_epochs, batch_size, lr, train_iter, test_iter, net, params, loss, updater):
    """线性分类模型训练实现——手动"""
    for epoch in range(num_epochs):
        # 模型训练
        metric = d2l_tools.Accumulator(3)  # 累计一个batch中损失总和、预测正确数量、总样本数
        for X, y in train_iter:
            # 1.正向传播
            y_hat = net(X, params[0], params[1])
            # 2.计算损失
            l = loss(y_hat, y)  # batch_size个元素的向量
            # 3.反向传播
            l.sum().backward()
            # 4.更新参数
            updater(params, lr, batch_size)
            # 5.累加损失和精度
            metric.add(float(l.sum()), evaluate.accuracy(y_hat, y), y.numel())  # 不参与backward
        # 模型评估
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]  # 训练完一个epoch后的平均损失、预测准确率
        test_acc = evaluate.evaluate_accuracy_scratch(net, test_iter, params[0], params[1])
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, train_acc {train_metrics[1]:.5f}, test_acc {test_acc:.5f}')
    print(f'完成训练\n')


def softmax_concise_train(num_epochs, train_iter, test_iter, net, loss, updater):
    """线性分类模型训练实现——自动"""
    for epoch in range(num_epochs):
        # 模型训练
        net.train()
        metric = d2l_tools.Accumulator(3)  # 累计一个batch中损失总和、预测正确数量、总样本数
        for X, y in train_iter:
            # 1.正向传播
            y_hat = net(X)
            # 2.计算损失
            l = loss(y_hat, y)
            # 3.反向传播
            updater.zero_grad()
            l.mean().backward()
            # 4.更新参数
            updater.step()
            # 5.累加损失和精度
            metric.add(float(l.sum()), evaluate.accuracy(y_hat, y), y.numel())  # 不累计梯度
        # 模型评估
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]  # 训练完一个epoch后的平均损失、预测准确率
        test_acc = evaluate.evaluate_accuracy_concise(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.5f}, train_acc {train_metrics[1]:.5f}, test_acc {test_acc:.5f}')
    print('完成训练')