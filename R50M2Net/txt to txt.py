# 打开文件并读取所有行
with open('predict0.txt', 'r') as f:
    lines = f.readlines()

# 移除空行
lines = [line for line in lines if line.strip() != ""]

# 将结果写回文件
with open('predict0.txt', 'w') as f:
    f.writelines(lines)

# 打开文件并读取所有行
with open('predict2.txt', 'r') as f:
    lines = f.readlines()

# 移除空行
lines = [line for line in lines if line.strip() != ""]

# 将结果写回文件
with open('predict2.txt', 'w') as f:
    f.writelines(lines)

    # 打开文件并读取所有行
    with open('predict1.txt', 'r') as f:
        lines = f.readlines()

    # 移除空行
    lines = [line for line in lines if line.strip() != ""]

    # 将结果写回文件
    with open('predict1.txt', 'w') as f:
        f.writelines(lines)