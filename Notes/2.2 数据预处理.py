import os

# %%
# 创建人工数据集并存储与CSV
os.makedirs(os.path.join('..', 'data'), exist_ok=True)      # 创建目录
data_file = os.path.join('..', 'data', 'house_tiny.csv')    # 新csv文件的路径字符串
with open(data_file, 'w') as f:
    f.write('NumRooms,RoofType,Price\n')
    f.write('NA,NA,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,Slate,178100\n')
    f.write('NA,NA,140000\n')


# %%
# 使用 pandas 读取数据
import pandas as pd
data = pd.read_csv(data_file)
print(data)


# %%
# 处理缺失数据 NA —— 插值/删除
inputs = data.iloc[:, 0:2]  # 除了最后一列
outputs = data.iloc[:, 2]   # 最后一列
print(inputs)
print(outputs)

inputs= inputs.fillna(inputs.mean()) # 数字取均值
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True) # 非数字根据类别取特征
print(inputs)


# %%
# 处理纯数字csv数据为张量tensor
import torch
X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(X)
print(y)