# %%
import torch


# %%
# 创建数组（形状/数据类型/值）
x1 = torch.arange(12)
print(x1)
print(x1.shape)          # 访问形状
print(x1.numel())        # 访问元素个数


# %%
# 改变元素形状
x2 = x1.reshape(3, 4)     # 注意x不会改变，只会返回一个新的torch
print(x2)
print(x2.shape)
print(x2.numel())


# %%
# 设定张量值
x3 = torch.zeros((2,3,4))       # 三维全0矩阵
x4 = torch.ones((2,3,4))        # 三维全1矩阵
x5 = torch.tensor([[1,2,3],[4,5,6]])    # 通过python列表直接赋值
print(x3)
print(x4)
print(x5)


# %%
# 标准算数运算（按对应元素进行）
x = torch.tensor([1,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
print(torch.exp(x))     # e^x


# %%
# 张量之间的运算
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1,3,4],[1,2,3,4],[3,4,5,6]])
print(x)
print(y)
print(torch.cat((x,y),dim=0))   # 在第0维合并
print(torch.cat((x,y),dim=1))   # 在第1维合并
print(x==y)     # 逻辑运算构建张量
print(x.sum())  # 返回一个标量


# %%
# 广播机制
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a)
print(b)
print(a+b)  # 维度相同，每一维进行复制


# %%
# 访问元素（元素/行/列/子区域/跳转子区域）
x = torch.arange(12).reshape((3,4))
print(x)
print(x[-1])    # 选择最后一个元素
print(x[1,2])
print(x[1,:])
print(x[:,1])
print(x[1:3, 1:2])
print(x[0::2,1])
x[0:2, :] = 12  # 多元素修改
print(x)


# %%
# 内存问题
x = torch.arange(12)
y = torch.zeros(12)
y = x + y       # 新变量
y[:] = x + y    # 原地复制
y += x          # 原地复制


# %%
# numpy转化
x = torch.arange(12).reshape((3,4))
A = x.numpy()       # 转化为 numpy
B = torch.tensor(A) # 转化为 tensor
print(type(A))
print(type(B))

y = torch.tensor([1.1])
print(y)
print(y.item())
print(float(y))
print(int(y))