# import os
# import numpy as np
# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
#
# def process_images(i):
#     i = str(i)
#
#     # 定义神经网络模型
#     model = nn.Sequential(
#         nn.Linear(1000, 500),
#         nn.ReLU(),
#         nn.Linear(500, 100),
#         nn.ReLU(),
#         nn.Linear(100, 3)
#     )
#
#     # 加载模型
#     model.load_state_dict(torch.load("128+200/1model.pth"))
#     model.eval()
#
#     # 定义一个函数来读取特征向量
#     def read_vector(file_path):
#         with open(file_path, "r", encoding="utf-8") as f:
#             vector = np.array(list(map(float, f.readline().split())))
#             return torch.tensor(vector, dtype=torch.float32)
#
#     # 读取两张图像
#     image1 = Image.open("C:/Users/34574\Desktop\op\post/post" +i+ ".jpg")
#     image2 = Image.open("C:/Users/34574\Desktop\op\pro/pro" +i+ ".jpg")
#
#     # 获取当前脚本的路径
#     script_path = os.path.dirname(os.path.realpath(__file__))
#
#     # 遍历三个文件夹中的所有文件
#     folder_paths = ["C:/Users/34574\Desktop\op\分类\+0/" +i, "C:/Users/34574\Desktop\op\分类\+1/" +i, "C:/Users/34574\Desktop\op\分类\+2/" +i]
#     for image, image_name in zip([image1, image2], ["post"+i+".jpg", "pro"+i+".jpg"]):
#         fig, ax = plt.subplots(1)
#         ax.imshow(image)
#
#         for folder_path in folder_paths:
#             for file in os.listdir(folder_path):
#                 if file.endswith(".txt"):
#                     file_path = os.path.join(folder_path, file)
#
#                     # 移除坐标轴
#                     ax.axis('off')
#
#                     # 读取特征向量并进行分类
#                     vector = read_vector(file_path)
#                     output = model(vector)
#                     predicted_class = torch.argmax(output).item()
#
#                     # 根据预测的类别选择颜色
#                     color = "green" if predicted_class == 2 else "yellow" if predicted_class == 1 else "red"
#
#                     # 从文件名中获取方框的坐标
#                     parts = file.split("_")
#                     coords = list(map(int, [parts[parts.index('p1')+1], parts[parts.index('p1')+2], parts[parts.index('p2')+1], parts[parts.index('p2')+2].split('.')[0]]))
#                     rect = patches.Rectangle((coords[0], coords[1]), coords[2]-coords[0], coords[3]-coords[1], linewidth=1, edgecolor=color, facecolor="none")
#
#                     # 在图像上添加方框
#                     ax.add_patch(rect)
#
#                     # 将文件名（无后缀）保存到对应类别的txt文件中
#                     with open(i+"+"+str(predicted_class) + ".txt", "a") as f:
#                         f.write(file.rsplit(".", 1)[0] + "\n")
#
#         # 移除图像周围的空白区域
#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#
#         # 保存图像
#         fig.savefig(os.path.join(script_path, image_name),dpi=300, bbox_inches='tight', pad_inches=0)
#
#     print("图像已保存")
#
# # 调用函数
# for a in range(22, 61):
#     process_images(a)
#
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def process_images(i):
    i = str(i)

    # 定义神经网络模型
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 3)
    )

    # 加载模型
    model.load_state_dict(torch.load("128+200/0.05model.pth"))
    model.eval()

    # 定义一个函数来读取特征向量
    def read_vector(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            vector = np.array(list(map(float, f.readline().split())))
            return torch.tensor(vector, dtype=torch.float32)

    # 读取两张图像
    image1 = Image.open("C:/Users/34574\Desktop\op\post/post" +i+ ".jpg")
    image2 = Image.open("C:/Users/34574\Desktop\op\pro/pro" +i+ ".jpg")

    # 获取当前脚本的路径
    script_path = os.path.dirname(os.path.realpath(__file__))

    # 遍历三个文件夹中的所有文件
    folder_paths = ["C:/Users/34574\Desktop\op\分类\+0/" +i, "C:/Users/34574\Desktop\op\分类\+1/" +i, "C:/Users/34574\Desktop\op\分类\+2/" +i]
    for image, image_name in zip([image1, image2], ["post"+i+".jpg", "pro"+i+".jpg"]):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for folder_path in folder_paths:
            if os.path.exists(folder_path) and os.listdir(folder_path):  # 检查文件夹是否存在且不为空
                for file in os.listdir(folder_path):
                    if file.endswith(".txt"):
                        file_path = os.path.join(folder_path, file)

                        # 移除坐标轴
                        ax.axis('off')

                        # 读取特征向量并进行分类
                        vector = read_vector(file_path)
                        output = model(vector)
                        predicted_class = torch.argmax(output).item()

                        # 根据预测的类别选择颜色
                        color = "green" if predicted_class == 2 else "yellow" if predicted_class == 1 else "red"

                        # 从文件名中获取方框的坐标
                        parts = file.split("_")
                        coords = list(map(int, [parts[parts.index('p1')+1], parts[parts.index('p1')+2], parts[parts.index('p2')+1], parts[parts.index('p2')+2].split('.')[0]]))
                        rect = patches.Rectangle((coords[0], coords[1]), coords[2]-coords[0], coords[3]-coords[1], linewidth=1, edgecolor=color, facecolor="none")

                        # 在图像上添加方框
                        ax.add_patch(rect)

                        # 将文件名（无后缀）保存到对应类别的txt文件中
                        with open(i+"+"+str(predicted_class) + ".txt", "a") as f:
                            f.write(file.rsplit(".", 1)[0] + "\n")

        # 移除图像周围的空白区域
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 保存图像
        fig.savefig(os.path.join(script_path, image_name),dpi=300, bbox_inches='tight', pad_inches=0)

    print("图像已保存")

# 调用函数
for a in range(1, 61):
    process_images(a)
#该程序有一个bug，生成的所有txt都是二倍的，复写了一遍，因此在计算混淆矩阵时，除以了2