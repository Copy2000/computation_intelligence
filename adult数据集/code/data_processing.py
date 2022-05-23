# import imp
# import pandas as pd
# import numpy as np
# MAX_ROW=32500
# file = pd.read_csv('D:\大三下\计算智能\第二次\代码及结果\\some.csv')
# # print(file)
# file=np.array(file[0:MAX_ROW])
# # 去除无效信息============================================================
# '''
# x=np.delete(x,1,axis=0) #二维数组删除1+1行
# 删除后结果：x=[[1,2,3],[7,8,9]]
# '''
# miss_data_row=[]
# for row in range(MAX_ROW):
#     if file[row][13]==" ?":
#         miss_data_row.append(row)
# file=np.delete(file,miss_data_row,axis=0)
# '''
# feature
# 1-5     age             workclass       fnlwgt          education       education-num
# 6-10    marital-status  occupation      relationship    race            sex
# 11-15   capital-gain    capital-loss    hours-per-week  native-country  class(target)
# '''
# # 获取属性名称=======================================================
# attribute_name=[]
# #   workclass  education   marital-status  occupation  relationship    race    sex     native-country  class
# index_=[1,3,5,6,7,8,9,13,14]
# for i in index_:
#     a=[]
#     for item in file[:,i]:
#         if item not in(a):
#             a.append(item)
#     # print(len(a))
#     # print()
#     attribute_name.append(a)
# # print(attribute_name)
# # 将属性变成数字==============================================
# for ats in attribute_name:
#     for flag in range(len(ats)):
#         pos=np.argwhere(file==ats[flag])
#         for i in pos:
#             x_=i[0]
#             y_=i[1]
#             file[x_,y_]=flag+1

# # 保存到some.csv==============================================================
# import csv
# with open('attribute.csv', 'w',newline="") as f:     
#     # 实例化csv.writer对象
#     writer = csv.writer(f)
#     # 用writerows方法将数据以指定形式保存
#     writer.writerows(attribute_name)

# with open('some1.csv', 'w',newline="") as f:     
#     # 实例化csv.writer对象
#     writer = csv.writer(f)
#     # 用writerows方法将数据以指定形式保存
#     writer.writerows(file)


# 删除capital-gain 和 capital-loss 和 fnlwgt 的数据=================
import csv
import imp
import pandas as pd
import numpy as np

col=[2,10,11]
file = pd.read_csv('D:\大三下\计算智能\第二次\代码及结果\\processed_adult.csv')
file=np.array(file)
file=np.delete(file,col,axis=1)
with open('processed_adult2.csv', 'w',newline="") as f:     
    # 实例化csv.writer对象
    writer = csv.writer(f)
    # 用writerows方法将数据以指定形式保存
    writer.writerows(file)

