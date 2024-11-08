data = [0.3766,0.576198,0.466984,0.549836,0.47075
2279.71,3442.3621,3009.2172,2393.6955,2576.0723
0.0482,0.242446,0.094954,0.06989,0.076156
0.0059,0.04602,0.01475,0.010915,0.011918
0.3594,0.50316,0.470814,0.44925,0.420498
0.0869,0.18249,0.133826,0.142516,0.125136
0.589,0.80693,0.64201,0.64201,0.589
0.1589,0.498946,0.309855,0.30191,0.227227
9140.68,7403.9508,17550.1056,10786.0024,11913.35293
1444121.81,2714949.003,3826922.797,2252830.024,2931567.274
2682087.62,1448327.315,4184056.687,2333416.229,2655266.744
2682088,5444638.64,6892966.16,4532728.72,5623444.507
13105.58,31977.6152,18347.812,18347.812,18872.0352
6952.63,18772.101,8134.5771,7925.9982,8343.156
27080.5,44412.02,61472.735,46849.265,50911.34
302582.83,928929.2881,995497.5107,571881.5487,832102.7825
38317.83,79701.0864,116103.0249,47514.1092,81106.0735
1136.17,2692.7229,3010.8505,1806.5103,2503.361233
9044.17,10943.4457,33825.1958,16460.3894,20409.67697
];
[Y,PS] = mapminmax(data',0,1);%由于此函数是按行进行归一化的，因此先转置再转回来就好了
to_one = Y';

ele_weight = [];
sum_col = sum(to_one); %默认按列求和
[row, col] = size(to_one); %获取原数据矩阵的行和列
for i = 1:row
    for j = 1:col
        ele_weight(i,j) = to_one(i,j)/sum_col(j); %计算出归一化后每个元素在所在特征列的占比
    end
end

E_ele = [];
for i = 1:row
    for j = 1:col
        if ele_weight(i,j) == 0 %规定0*ln(0) = 0,不赋值默认为0
            continue
        end
        E_ele(i,j) = -ele_weight(i,j)*log(ele_weight(i,j));%计算信息熵
    end
end

E = sum(E_ele./log(row));%计算此特征的信息熵
sum_E = sum(E);
W = (1-E)./(col-sum_E);%通过信息熵计算对应特征的权重
W = W';%转置便于矩阵乘法直接计算出对应的评价分数

data * W