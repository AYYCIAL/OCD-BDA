import os
import shutil

# 源文件夹和目标文件夹
source_directory = 'result/result0.05'
target_directory = 'txt3/model0.05/0'

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_directory):
    # 检查文件是否为.txt文件并且文件名（不包括后缀）以"1"结尾
    if filename.endswith('.txt') and filename[:-4].endswith('0'):
        # 构造完整的文件路径
        source = os.path.join(source_directory, filename)
        target = os.path.join(target_directory, filename)

        # 移动文件到目标文件夹
        shutil.move(source, target)

print("文件移动完成！")
