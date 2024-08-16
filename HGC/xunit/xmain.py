from xunit.unit import predict
from xunit.unit import cutpic
from xunit.unit import organize_images
from xunit.unit import asm
from xunit.unit import txt2excel
from xunit.unit import twoto1
from xunit.unit import normalize_excel_column
from xunit.unit import myhierarchical
from xunit.unit import copy_folder
from xunit.unit import delete_file
from xunit.unit import count_label_occurrences
from xunit.unit import select_rows_by_filenames
from xunit.unit import myGauss

def main(proimage_path, postimage_path):
    # 检测相关变量
    offset_x = 0
    offset_y = 0  # 用于图像分割的偏移量

    # 内部变量，一般不需要修改
    file_path = 'p1p2.txt'
    output_file_path = 'p1p2+.txt'
    output_folder_path_pro = 'pic'
    output_folder_path_post = 'pic+'

    # 调用 predict 函数进行预测，将会存储一个预测结果图与坐标框txt
    print('step1 predict:')
    predict(postimage_path)
    postcount = count_label_occurrences('p1p2.txt', 'label: Building')
    delete_file("p1p2.txt")
    predict(proimage_path)
    procount = count_label_occurrences('p1p2.txt', 'label: Building')
    if postcount < procount / 2:
        aaa = 0  # 代表着阳性样本较多
    else:
        aaa = 1  # 代表着阴性样本较多
    print(aaa, postcount, procount)
    print('\n step1 over')

    # 利用step1生成的p1p2.txt进行图像分割
    print('step2 cutpic:')
    cutpic(file_path, output_file_path, output_folder_path_pro, proimage_path, 0, 0)
    cutpic(file_path, output_file_path, output_folder_path_post, postimage_path, offset_x, offset_y)
    copy_folder('pic+', 'PIC++')
    print('\n step2 over')

    # 将step2生成的分割好的样本存储在子文件夹中，方便计算其特征
    print('step3 organize_images:')
    organize_images(output_folder_path_pro)
    organize_images(output_folder_path_post)
    print('\n step3 over')

    # 计算各个子文件夹内图片的特征，并储存为txt
    print('step4 asm:')
    asm(output_folder_path_pro)
    asm(output_folder_path_post)
    print('\n step4 over')

    # 将子文件夹中零散的特征文件整理为一个表格
    print('step5 txt2excel:')
    excel_file_path_pro = "data.xlsx"
    excel_file_path_post = "data+.xlsx"
    txt2excel(output_folder_path_pro, excel_file_path_pro)
    txt2excel(output_folder_path_post, excel_file_path_post)
    print('\n step5 over')

    # 归一化
    print('step6 to1:')
    normalize_excel_column("data+.xlsx", "ASM")
    normalize_excel_column("data+.xlsx", "Contrast")
    normalize_excel_column("data+.xlsx", "Correlation")
    normalize_excel_column("data+.xlsx", "Entropy")
    normalize_excel_column("data+.xlsx", "Dissimilarity")
    normalize_excel_column("data+.xlsx", "IDM")

    normalize_excel_column("data.xlsx", "ASM")
    normalize_excel_column("data.xlsx", "Contrast")
    normalize_excel_column("data.xlsx", "Correlation")
    normalize_excel_column("data.xlsx", "Entropy")
    normalize_excel_column("data.xlsx", "Dissimilarity")
    normalize_excel_column("data.xlsx", "IDM")
    print('\n step6 over')

    # 将两个归一化后的特征表格相减
    print('step7 twoto1:')
    twoto1()
    print('\n step7 over')

    # H算法，获取阴性样本
    print('step8 myhierarchical:')
    myhierarchical()
    print('\n step8 over')

    # 根据阳性样本占总样本比例，选择需要高斯分类的文件夹
    print('step9 myGauss:')
    if aaa == 0:
        select_rows_by_filenames("dataresult.xlsx", "result+", "results+.xlsx")
        myGauss('results+.xlsx', 'Clusterresults+.xlsx')
        print('结果保存在:result,result-,result--')
        with open('111AAA.txt', 'w') as file:
            file.write("结果保存在:result,result-,result--")
    else:
        select_rows_by_filenames("dataresult.xlsx", "result", "results.xlsx")
        myGauss('results.xlsx', 'Clusterresults.xlsx')
        print('结果保存在:result+,result-,result--')
        with open('111AAA.txt', 'w') as file:
            file.write("结果保存在:result+,result-,result--")
    # result-为样本文件,result--为样本文件
    print('\n step9 over')

# 示例调用
if __name__ == "__main__":
    proimage_path = 'pro.jpg'
    postimage_path = 'post.jpg'
    main(proimage_path, postimage_path)
