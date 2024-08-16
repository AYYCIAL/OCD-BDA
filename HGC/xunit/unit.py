from PIL import Image
from ultralytics import YOLO
import os
import shutil
import numpy as np
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from openpyxl import Workbook
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def predict(image_path):
    # 加载预训练的YOLOv8n模型
    model = YOLO('C:\yolov8/ultralytics-main/ultralytics-main/runs\detect/train84-2-400\weights/best.pt')

    # 在指定图像上运行推理
    results = model(image_path)  # 结果列表

    # 展示结果
    for r in results:
        im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        im.save('proresult.jpg')  # 保存图像

def cutpic(file_path, output_file_path, output_folder_path, image_path, offset_x=0, offset_y=0):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i in range(2, len(lines), 3):
            current_line = lines[i].strip()
            if current_line.startswith("label: Building"):
                extracted_content = ''.join(lines[i - 2:i])
                output_file.write(extracted_content)

    print("提取的内容已写入：", output_file_path)

    os.makedirs(output_folder_path, exist_ok=True)

    with open(output_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        p1_coordinates = eval(lines[i].split(': ')[1])
        p2_coordinates = eval(lines[i + 1].split(': ')[1])

        p1_coordinates = (p1_coordinates[0] + offset_x, p1_coordinates[1] + offset_y)
        p2_coordinates = (p2_coordinates[0] + offset_x, p2_coordinates[1] + offset_y)

        image = Image.open(image_path)

        cropped_region = image.crop((p1_coordinates[0], p1_coordinates[1], p2_coordinates[0], p2_coordinates[1]))

        output_image_name = f'image_{i//2 + 1}_p1_{p1_coordinates[0]}_{p1_coordinates[1]}_p2_{p2_coordinates[0]}_{p2_coordinates[1]}.jpg'
        output_image_path = os.path.join(output_folder_path, output_image_name)

        cropped_region.save(output_image_path)

        print(f"截取的矩形区域 {i//2 + 1} 已保存到：", output_image_path)

def organize_images(folder_path):
    # 检查给定的文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("文件夹路径不存在！")
        return

    # 获取文件夹中所有图片文件的列表
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("文件夹中没有图片文件！")
        return

    # 在文件夹下创建子文件夹，并将图片移动到相应的子文件夹中
    for image_file in image_files:
        # 提取图片文件名（不含扩展名）
        image_name = os.path.splitext(image_file)[0]
        # 创建与图片同名的子文件夹
        subfolder_path = os.path.join(folder_path, image_name)
        os.makedirs(subfolder_path, exist_ok=True)
        # 移动图片文件到相应的子文件夹中
        image_file_path = os.path.join(folder_path, image_file)
        shutil.move(image_file_path, subfolder_path)

    print("图片已经成功整理到相应的子文件夹中！")

def asm(folder_path):
    # 获取文件夹下所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 循环处理每个子文件夹中的图片
    for subfolder in subfolders:
        print(f"处理子文件夹: {subfolder}")
        # 获取子文件夹中所有图片文件的路径
        image_files = [f.path for f in os.scandir(subfolder) if
                       f.is_file() and f.name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # 循环处理每张图片
        for img_path in image_files:
            print(f"  处理图片: {img_path}")
            # 读取图像
            img = io.imread(img_path)
            # 如果图像是彩色的，转换为灰度图像
            if len(img.shape) > 2:
                img = color.rgb2gray(img)
            # 确保图像是8位整数灰度图像
            img = (img * 255).astype(np.uint8)
            # 计算灰度共生矩阵
            glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            # 计算纹理特征
            asm = graycoprops(glcm, 'ASM')[0, 0]
            con = graycoprops(glcm, 'contrast')[0, 0]
            cor = graycoprops(glcm, 'correlation')[0, 0]
            ent = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
            dis = graycoprops(glcm, 'dissimilarity')[0, 0]
            idm = graycoprops(glcm, 'homogeneity')[0, 0]
            # 构建保存结果的文件路径
            result_file = os.path.splitext(img_path)[0] + "_texture_features.txt"
            # 将结果保存到文件中
            with open(result_file, 'w') as f:
                f.write(
                    f'ASM: {asm}, Contrast: {con}, Correlation: {cor}, Entropy: {ent}, Dissimilarity: {dis}, IDM: {idm}')


def txt2excel(folder_path, excel_file_path):
    # 创建一个新的 Excel 工作簿
    wb = Workbook()
    # 激活第一个工作表
    ws = wb.active
    # 设置表头
    ws.append(["文件名", "ASM", "Contrast", "Correlation", "Entropy", "Dissimilarity", "IDM"])

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, "r") as f:
                    data = f.read().strip().split(", ")
                    file_name = os.path.basename(file_path)
                    # 将数据写入 Excel 表格
                    ws.append([file_name] + [float(item.split(": ")[1]) for item in data])

    # 保存 Excel 文件
    wb.save(excel_file_path)

def twoto1():
    # 读取第一个Excel文件
    file1 = pd.read_excel("data+.xlsx")

    # 读取第二个Excel文件
    file2 = pd.read_excel("data.xlsx")

    # 将两个文件的对应列相减
    result = pd.DataFrame()
    result['文件名'] = file1['文件名']  # 复制第一列内容到结果
    result['ASM'] = file1['ASM'] - file2['ASM']
    result['Contrast'] = file1['Contrast'] - file2['Contrast']
    result['Correlation'] = file1['Correlation'] - file2['Correlation']
    result['Entropy'] = file1['Entropy'] - file2['Entropy']
    result['Dissimilarity'] = file1['Dissimilarity'] - file2['Dissimilarity']
    result['IDM'] = file1['IDM'] - file2['IDM']

    # 将结果写入新的Excel文件
    result.to_excel("dataresult.xlsx", index=False)

    # 读取dataresult.xlsx文件
    dataresult = pd.read_excel("dataresult.xlsx")

    # 将'_texture_features.txt'内容替换为空白
    dataresult.replace(to_replace='_texture_features.txt', value='', inplace=True, regex=True)

    # 将修改后的结果重新写入dataresult.xlsx文件
    dataresult.to_excel("dataresult.xlsx", index=False)

def normalize_column(column):
    max_val = max(column)
    min_val = min(column)
    if max_val == min_val:
        return [0.5] * len(column)  # 如果所有值相同，则对所有值返回0.5
    else:
        return [(x - min_val) / (max_val - min_val) for x in column]


def normalize_excel_column(excel_file, column_name):
    # 读取 Excel 表格数据
    df = pd.read_excel(excel_file)

    # 检查列是否存在
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the Excel file.")
        return

    # 获取指定列数据
    column_data = df[column_name]

    # 归一化处理
    normalized_data = normalize_column(column_data)

    # 更新 DataFrame 中指定列的数据
    df[column_name] = normalized_data

    # 将归一化后的数据写回 Excel 文件
    with pd.ExcelWriter(excel_file) as writer:
        df.to_excel(writer, index=False)

def create_folder(folder_name):
    try:
        os.mkdir(folder_name)
        print("Folder '{}' created successfully.".format(folder_name))
    except FileExistsError:
        print("Folder '{}' already exists.".format(folder_name))
    except Exception as e:
        print("An error occurred: ", str(e))

def myhierarchical():
    # 读取Excel文件
    data = pd.read_excel('dataresult.xlsx')

    # 选择需要用来聚类的列
    selected_columns = ['ASM', 'Dissimilarity', 'IDM']
    X = data[selected_columns]

    # 初始化Hierarchical Clustering模型并指定聚类数量为2
    hierarchical = AgglomerativeClustering(n_clusters=2)

    # 对数据进行聚类
    labels = hierarchical.fit_predict(X)

    # 将聚类结果添加到原始数据中
    data['Cluster'] = labels

    # 将结果保存到Excel文件中
    data.to_excel('result_with_cluster_hierarchical.xlsx', index=False)

    # 读取包含聚类结果的Excel文件
    data = pd.read_excel('result_with_cluster_hierarchical.xlsx')

    create_folder('result')
    create_folder('result+')
    # 遍历每一行数据
    for index, row in data.iterrows():
        # 获取文件名和聚类结果
        filename = row['文件名']+'.jpg'
        cluster = row['Cluster']

        # 构建目标文件夹路径
        if cluster == 0:
            destination_folder = 'result'
        else:
            destination_folder = 'result+'

        # 拷贝文件到目标文件夹
        source_path = os.path.join('PIC++', filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copyfile(source_path, destination_path)


def copy_folder(source_folder, destination_folder):
    try:
        # 创建目标文件夹
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # 遍历源文件夹中的所有文件和子文件夹
        for item in os.listdir(source_folder):
            source_item = os.path.join(source_folder, item)
            destination_item = os.path.join(destination_folder, item)

            # 如果是文件，则直接复制
            if os.path.isfile(source_item):
                shutil.copy2(source_item, destination_item)

            # 如果是文件夹，则递归调用copy_folder函数
            elif os.path.isdir(source_item):
                copy_folder(source_item, destination_item)

        print("文件夹复制成功！")

    except Exception as e:
        print("文件夹复制失败:", e)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 '{file_path}' 已删除。")
    else:
        print(f"文件 '{file_path}' 不存在。")

def count_label_occurrences(file_path, label):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if label in line:
                count += 1
    return count

def select_rows_by_filenames(original_file_path, result_folder, output_file_path):
    """
    从原始Excel文件中提取符合条件的行，并将其写入新的Excel文件。

    Parameters:
        original_file_path (str): 原始Excel文件的路径。
        result_folder (str): 存放结果文件的文件夹路径。
        output_file_path (str): 输出文件的路径。

    Returns:
        None
    """
    # 读取result文件夹中的所有文件名（不含后缀）
    result_files = [os.path.splitext(file)[0] for file in os.listdir(result_folder)]

    # 读取原始Excel文件
    df = pd.read_excel(original_file_path)

    # 提取符合条件的行
    selected_rows = df[df['文件名'].apply(lambda x: os.path.splitext(x)[0] in result_files)]

    # 将符合条件的行写入新的Excel文件
    selected_rows.to_excel(output_file_path, index=False)

def myGauss(input_file_path, output_file_path):
    """
    读取Excel文件，对指定列进行高斯混合模型聚类，将聚类结果添加到原始数据中，并根据聚类结果移动文件到不同的文件夹中。

    Parameters:
        input_file_path (str): 输入Excel文件的路径。
        output_file_path (str): 输出Excel文件的路径。

    Returns:
        None
    """
    # 读取Excel文件
    data = pd.read_excel(input_file_path)

    # 选择需要用来聚类的列
    selected_columns = ['ASM', 'Dissimilarity', 'IDM']
    X = data[selected_columns]

    # 初始化高斯混合模型并指定聚类数量为2
    gmm = GaussianMixture(n_components=2)

    # 对数据进行聚类
    gmm.fit(X)

    # 获取聚类结果
    labels = gmm.predict(X)

    # 将聚类结果添加到原始数据中
    data['Cluster'] = labels

    # 将结果保存到Excel文件中
    data.to_excel(output_file_path, index=False)

    # 读取包含聚类结果的Excel文件
    data = pd.read_excel(output_file_path)

    create_empty_folder("result-")
    create_empty_folder("result--")

    # 遍历每一行数据
    for index, row in data.iterrows():
        # 获取文件名和聚类结果
        filename = row['文件名']+'.jpg'
        cluster = row['Cluster']

        # 构建目标文件夹路径
        if cluster == 0:
            destination_folder = 'result-'
        else:
            destination_folder = 'result--'

        # 拷贝文件到目标文件夹
        source_path = os.path.join('PIC++', filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copyfile(source_path, destination_path)

def create_empty_folder(folder_path):
    """
    创建一个空白文件夹。

    Parameters:
        folder_path (str): 要创建的文件夹路径。

    Returns:
        None
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 创建成功！")
    except OSError as e:
        print(f"创建文件夹 '{folder_path}' 时出错：{e}")