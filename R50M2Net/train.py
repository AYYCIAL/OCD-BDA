def train_model(abc):
    import os
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # 定义文件夹路径
    base_path = "C:/Users/34574/Desktop/op/分类"
    folders = ["+0", "+1", "+2"]

    # 定义数据集
    class VectorDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # 加载数据
    data = []
    labels = []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(base_path, folder)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(subfolder_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        vector = np.array(list(map(float, f.readline().split())))
                        data.append(vector)
                        labels.append(label)

    # 划分训练集和测试集
    train_ratio = abc  # 训练样本占总样本的比例
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_ratio, random_state=None)

    # 创建数据集和数据加载器
    train_dataset = VectorDataset(train_data, train_labels)
    test_dataset = VectorDataset(test_data, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # 定义神经网络模型
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 3)
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练模型
    losses = []  # 用于记录每个epoch的loss
    for epoch in range(200):  # 这里我们只训练200轮，你可以根据需要调整
        for vectors, labels in train_dataloader:
            outputs = model(vectors)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        losses.append(loss.item())  # 记录每个epoch的loss

    abc=str(abc)

    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(abc + "loss_curve.png")  # 保存loss曲线为png图片

    # 保存模型
    torch.save(model.state_dict(), abc + "model.pth")

    print("模型训练完成并保存为model.pth，loss曲线已保存为loss_curve.png")

# 调用函数
train_model(0.6)
# import numpy as np
#
# for a in np.arange(0.05, 1, 0.05):
#     train_model(a)