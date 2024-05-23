import random
import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def synthetic_data(w, b, num_examples):
    """生成数据——线性回归"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter_by_hand(batch_size, features, labels):
    """读取数据——手动实现"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def load_array(data_arrays, batch_size, is_train=True):
    """读取数据——自动实现"""
    dataset = data.TensorDataset(*data_arrays)  # 转化类型
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回迭代器


def get_dataloader_workers():
    """读取数据的进程数"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """读取数据——Fashion-MNIST"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))