import os

# 指定你要整合的文件夹
folder_path = 'txt3/model0.05/2'  # 请将这里替换为你的文件夹路径
# 指定输出的文件名
output_file = 'predict2.txt'  # 你可以根据需要修改这个文件名

# 创建一个空的列表来保存所有的文本
texts = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是txt文件
    if filename.endswith('.txt'):
        # 打开并读取文件
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
        # 将文本添加到列表中
        texts.append(text)

# 将所有的文本写入到一个新的文件中
with open(output_file, 'w') as f:
    for text in texts:
        f.write(text + '\n')
