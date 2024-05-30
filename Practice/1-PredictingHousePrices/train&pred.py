import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#%%
# 读取并合并数据
train_data = pd.read_csv('../../data/kaggle_house_pred/train.csv')
test_data = pd.read_csv('../../data/kaggle_house_pred/test.csv')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print('所有加载数据完成：', all_features.shape, '\n', all_features.iloc[0:4, [0,1,2,-2,-1]])


# 数据处理
# 1.数值（Z得分标准化）
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
mean_values = all_features[numeric_features].mean() # 计算每个数值特征的均值
all_features[numeric_features] = all_features[numeric_features].fillna(mean_values) # 缺失值填充均值
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std())) # 标准化
print('数值类型标准化完成', all_features.shape, '\n', all_features.iloc[0:4, [0,1,2,-2,-1]])

# 2.类别（独热编码）
all_features = pd.get_dummies(all_features, dummy_na=True)  # 创建指示符特征
print('类别编码完成', all_features.shape, '\n', all_features.iloc[0:4, [1, 2, 3, -3, -2,-1]])


# 还原数据集
train_num = train_data.shape[0]
train_features = torch.tensor(all_features[:train_num].values, dtype=torch.float32)
test_features = torch.tensor(all_features[train_num:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


# 获取k折数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


#%%
# 训练
def log_rmse(net, loss, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

# 训练一次（跑几个epoch）
def train(net, loss, optimizer,
          train_features, train_labels, test_features, test_labels,
          num_epochs, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, loss, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, loss, test_features, test_labels))
    return train_ls, test_ls


#%%
# 验证
def valid_k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # net = nn.Sequential(nn.Linear(train_features.shape[1], 1))
        net = nn.Sequential(
            nn.Linear(train_features.shape[1], 32),  # 第一个隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout层
            nn.Linear(32, 1)  # 输出层
        )
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 训练一折
        train_ls, valid_ls = train(net, loss, optimizer,
                                   *data,
                                   num_epochs,  batch_size)
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 'f'验证log rmse{float(valid_ls[-1]):f}')
        # 绘图可视化（还没绘）
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
    return train_l_sum / k, valid_l_sum / k


#%%
# 决策模型超参数
# 使用k折交叉验证得到的性能数据来做出决策：
#   1.如果看到某个参数配置明显优于其他配置，选择这个配置来进行全数据集的训练。
#   2.如果性能没有显著差异，选择更简单或更快的模型配置。
k, num_epochs, lr, weight_decay, batch_size = 5, 200, 3, 0, 64
train_l, valid_l = valid_k_fold(k, train_features, train_labels, num_epochs, lr,weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')


#%%
# 预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = nn.Sequential(nn.Linear(train_features.shape[1], 1))
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_ls, _ = train(net, loss, optimizer,
                        train_features, train_labels, None, None,
                        num_epochs, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 重新格式化以导出
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# 确定好合适地超参数后即可运行
# train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)