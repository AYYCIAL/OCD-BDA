#result为阳性，result+为阴性，result--为距离他们最远的分类；高斯崩溃是某一预测量过少，进入下一个循环即可
import os
from xunit.xmain import main

for i in range(29, 61): #后一个数取到样本量+1
    # 创建存储文件的文件夹
    folder_name = str(i)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 图片路径
    proimage_path = f'C:/Users/34574/Desktop/op/pro/pro{i}.jpg'
    postimage_path = f'C:/Users/34574/Desktop/op/post/post{i}.jpg'

    # 在指定的文件夹中运行主程序
    os.chdir(folder_name)
    main(proimage_path, postimage_path)
    os.chdir('..')  # 返回上一级目录，以便下一次循环创建新的文件夹
