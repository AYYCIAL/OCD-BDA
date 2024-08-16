import os
import numpy as np

# 定义文件夹路径
base_path = "C:/Users/34574/Desktop/op/分类"
minus_zero_path = os.path.join(base_path, "-2")
zero_path = os.path.join(base_path, "2")
plus_zero_path = os.path.join(base_path, "+2")

# 创建+0文件夹
os.makedirs(plus_zero_path, exist_ok=True)

# 遍历-0文件夹下的子文件夹
for folder in os.listdir(minus_zero_path):
    minus_zero_subfolder_path = os.path.join(minus_zero_path, folder)
    zero_subfolder_path = os.path.join(zero_path, folder)
    plus_zero_subfolder_path = os.path.join(plus_zero_path, folder)

    # 创建+0下的同名子文件夹
    os.makedirs(plus_zero_subfolder_path, exist_ok=True)

    # 遍历子文件夹下的txt文件
    for file in os.listdir(minus_zero_subfolder_path):
        if file.endswith(".txt"):
            minus_zero_file_path = os.path.join(minus_zero_subfolder_path, file)
            zero_file_path = os.path.join(zero_subfolder_path, file)
            plus_zero_file_path = os.path.join(plus_zero_subfolder_path, file)

            # 读取-0和0文件夹下的同名txt文件中的特征向量
            with open(minus_zero_file_path, "r", encoding="utf-8") as f:
                minus_zero_vector = np.array(list(map(float, f.readline().split())))

            with open(zero_file_path, "r", encoding="utf-8") as f:
                zero_vector = np.array(list(map(float, f.readline().split())))

            # 计算差值并归一化
            diff_vector = minus_zero_vector - zero_vector
            normalized_vector = diff_vector / np.linalg.norm(diff_vector)

            # 将结果保存在+0文件夹下的同名txt文件中
            with open(plus_zero_file_path, "w", encoding="utf-8") as f:
                f.write(" ".join(map(str, normalized_vector)))
