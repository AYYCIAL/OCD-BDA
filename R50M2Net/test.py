import numpy as np
import torch
from torch import nn

# 定义神经网络模型
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 3)
)

# 加载模型
model.load_state_dict(torch.load("50model.pth"))
model.eval()

# 定义一个函数来读取特征向量
def read_vector(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        vector = np.array(list(map(float, f.readline().split())))
        return torch.tensor(vector, dtype=torch.float32)

# 读取要分类的特征向量
vector = read_vector("2.txt")

# 使用模型进行分类
output = model(vector)
predicted_class = torch.argmax(output).item()

print(f"预测的类别是：{predicted_class}")
