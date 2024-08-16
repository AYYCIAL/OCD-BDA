import os
from PIL import Image

# 定义源文件夹和目标文件夹
src_folder = "C:\\Users\\34574\\Desktop\\op\\分类\\2"
pro_folder = "C:\\Users\\34574\\Desktop\\op\\pro"
dst_folder = "C:\\Users\\34574\\Desktop\\op\\分类\\-2"

# 遍历源文件夹
for folder_name in os.listdir(src_folder):
    # 找到对应的pro图片
    pro_image_path = os.path.join(pro_folder, "pro" + folder_name + ".jpg")
    pro_image = Image.open(pro_image_path)

    # 遍历文件夹中的图片
    folder_path = os.path.join(src_folder, folder_name)
    for image_name in os.listdir(folder_path):
        # 提取p1和p2的坐标
        parts = image_name.split("_")
        p1_x, p1_y = int(parts[3]), int(parts[4])
        p2_x, p2_y = int(parts[6]), int(parts[7].split(".")[0])

        # 在pro图片上框出矩形区域并截图
        cropped_image = pro_image.crop((p1_x, p1_y, p2_x, p2_y))

        # 在目标文件夹下创建对应的文件夹
        dst_subfolder = os.path.join(dst_folder, folder_name)
        os.makedirs(dst_subfolder, exist_ok=True)

        # 保存截图
        cropped_image_path = os.path.join(dst_subfolder, image_name)
        cropped_image.save(cropped_image_path)
