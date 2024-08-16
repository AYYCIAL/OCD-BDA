
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 加载预训练的transformer模型
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model.eval()

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 图片文件夹路径
image_folder_path = "C:/Users/34574/Desktop/op/分类"

# 遍历文件夹及其子文件夹中的所有图片
for root, dirs, files in os.walk(image_folder_path):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 可以根据需要添加其他图片格式
            # 图片路径
            image_path = os.path.join(root, filename)

            # 加载图片
            image = Image.open(image_path)

            # 预处理图片
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)

            # 确保不会计算梯度
            with torch.no_grad():
                # 提取特征
                feature_vector = model(input_batch)

            # 将特征向量转换为numpy数组
            feature_vector_np = feature_vector.numpy()

            # 获取图片的文件名（不包括扩展名）
            base_name = os.path.splitext(filename)[0]

            # 将特征向量保存到与图片同名的TXT文件
            np.savetxt(os.path.join(root, f'{base_name}.txt'), feature_vector_np)

            # 打印特征向量的形状
            print(f"特征向量的形状（{filename}）：", feature_vector.shape)
