import pandas as pd
import numpy as np
import math
from numpy import array
import copy

# 熵值法函数 求指标权重
def cal_entropy_weight(x, label = dict()):
    '''熵值法计算变量的权重'''
    print('原始数据：'+ str(x))
    # 获得表的列头
    x_header = x.keys()
    # 初始化默认的label：1为正向指标 2 为负向指标
    if len(label) == 0:
        for key in x_header:
            label[key] = 1
    print('label:'+str(label))
    # 标准化
    for key in label:
        # 得到列最大和最小
        maxium = np.max(x[key], axis=0)
        minium = np.min(x[key], axis=0)
        # print('maxium:' + str(maxium))
        # print('minium:' + str(minium))
        if label[key] == 1:# 正向指标
            x[key] = x.loc[:, key].apply(lambda x: ((x - minium) / (maxium - minium)))
            # x[key].apply(lambda x: ((x - maxium) / (np.max(x) - np.min(x))))
        elif label[key] == 2: # 负向指标
            x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium)))
        else:
            print('输入的数据标签有误！')
            return
        # x_standardized.append(pd.DataFrame({key:temp_x}), ignore_index=True)

    # print('x_standardized:' + str(x))
    # 真向标准化
    # x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    # 负向标准化
    # x = x.apply(lambda x: ((np.max(x) - x) / (np.max(x) - np.min(x))))
    # 获得标准化的表中的值
    x_standardized_data = list(x.to_dict().values())
    print('标准化：' + str(x))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)
    lnf = [[None] * cols for i in range(rows)]
    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    # p_arry = array()
    temp_p_list = list()
    p_row_list = list()
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
                p_row_list.append(0.0000)
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                # print(type(p))
                p_row_list.append(p)
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
        temp_p_list.append(copy.deepcopy(p_row_list))
        p_row_list.clear()
    # 行列转置
    p_list = [[r[col] for r in temp_p_list] for col in range(len(temp_p_list[0]))]
    p_dict = dict()
    p_num = 0
    for k in x_header:
        p_dict[k] = p_list[p_num]
        p_num += 1
    p_df = pd.DataFrame(p_dict)
    # print(p_list)
    print('参数的熵' + str(p_df))
    lnf = pd.DataFrame(lnf)
    E = lnf
    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
     # 计算各样本的综合得分,用标准化后的数据
    record_row = 0.0
    record = list()
    record_sum = 0.00
    x_standardized_data =  [[r[col] for r in x_standardized_data] for col in range(len(x_standardized_data[0]))]
    # print('转置后的标准化:'+str(x_original_data))
    for i in range(len(x_standardized_data)):
        for j in range(len(x_standardized_data[i])):
            # print('x_original_data[i][j]' + str(x_original_data[i][j]))
            # print('w:' + str(w[i]))
            record_sum += x_standardized_data[i][j]*w[j]
            record_row += x_standardized_data[i][j]*w[j]
        # print('record_row:' + str(round(record_row, 2)))
        record.append(round(record_row * 100, 2))
        record_row = 0.0
    print('record:' + str(record))
    record_average = round(record_sum/len(record)*100, 2)
    print('record_average:' + str(record_average))
    w = pd.DataFrame(w)
    return w


a = {}
b = {}
c = {}
d = {}

deal_b= dict()
for key, value in b.items():
    deal_b[key] = abs(b[key] - 1)
data = {'##':list(a.values()),}
label = {'##':1,}

# 求熵权法
df = pd.DataFrame(data)
df.dropna()
w = cal_entropy_weight(df, label)  # 调用cal_weight
w.index = df.columns
w.columns = ['weight']
print(w)
