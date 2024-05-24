import torch
from d2l import torch as d2l


def get_fashion_mnist_labels(labels):
    """根据数字标签返回 Fashion-MNIST 数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):    # 张量图片
            ax.imshow(img.numpy())
        else:   # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class Accumulator:
    """对 n个变量求和"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        # 将data中的累积值和args传入的新值按位置配对
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # 解释：
        # 当self.data=[1,1], args=[3,5]时, zip=[(1,3),(1,5)];
        # 当for循环时, 每个元素是一个二元元组(a,b), a是原始累计值, b是传入新值
        # 最后使用列表推导 [a + float(b) for …] 组成一个新的数组，与原始数组大小相同

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs) # 通用类型转换函数
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)