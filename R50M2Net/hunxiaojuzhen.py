import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def count_matches(predict_file, gt_files):
    predict_lines = read_file(predict_file)
    gt_lines = [read_file(gt_file) for gt_file in gt_files]

    counts = [0, 0, 0]
    for line in predict_lines:
        for i in range(3):
            if line in gt_lines[i]:
                counts[i] += 1
    return [count / 2 for count in counts]  # 将计数除以2

gt_files = ['GT/GT0.txt', 'GT/GT1.txt', 'GT/GT2.txt']
predict_files = ['predict0.txt', 'predict1.txt', 'predict2.txt']

matrix = np.array([count_matches(predict_file, gt_files) for predict_file in predict_files])

# 计算精确度、召回率和F1分数
precision = np.diag(matrix) / np.sum(matrix, axis=0)
recall = np.diag(matrix) / np.sum(matrix, axis=1)
f1_score = 2 * precision * recall / (precision + recall)

# 计算特异度
FP = np.sum(matrix, axis=0) - np.diag(matrix)
TN = np.sum(matrix) - np.diag(matrix) - FP
specificity = TN / (FP + TN)

# 计算总体的精确度
total_precision = np.sum(np.diag(matrix)) / np.sum(matrix)

# 打印每个类别的指标
for i in range(3):
    print(f"类别{i}的精确度: {precision[i]:.2f}, 召回率: {recall[i]:.2f}, F1分数: {f1_score[i]:.2f}, 特异度: {specificity[i]:.2f}")

# 打印总体的精确度
print(f"\n总体精确度: {total_precision:.2f}")

# 打印混淆矩阵
print("\n混淆矩阵:")
print(matrix)
