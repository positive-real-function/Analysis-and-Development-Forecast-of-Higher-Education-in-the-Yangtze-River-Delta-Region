import pandas as pd

x = pd.read_excel('data.xlsx')
x = x.iloc[:, :].T
print(x)

# 1、数据均值化处理
x_mean = x.mean(axis=1)
for i in range(x.index.size):
    x.iloc[i, :] = x.iloc[i, :] / x_mean[i]

# 1、数据差值化处理
x = (x - x.min()) / (x.max() - x.min())
x = x.T

# 1、数据初值化处理
x_mean = x.mean(axis=1)
for i in range(x.index.size):
    x.iloc[i, :] = x.iloc[i, :] / x.iloc[i, 0]

# 2、提取参考队列和比较队列
ck = x.iloc[0, :]
print(" 参考队列：", ck)
cp = x.iloc[1:, :]
print(" 参考队列：", cp)

# 比较队列与参考队列相减
t = pd.DataFrame()
for j in range(cp.index.size):
    temp = pd.Series(cp.iloc[j, :] - ck)
    t = t.append(temp, ignore_index=True)

# 求最大差和最小差
mmax = t.abs().max().max()
mmin = t.abs().min().min()
rho = 0.4

# 3、求关联系数
ksi = ((mmin + rho * mmax) / (abs(t) + rho * mmax))

# 4、求关联度
r = ksi.sum(axis=1) / ksi.columns.size

# 5、关联度排序，得到结果
result = r.sort_values(ascending=False)

print(r)


