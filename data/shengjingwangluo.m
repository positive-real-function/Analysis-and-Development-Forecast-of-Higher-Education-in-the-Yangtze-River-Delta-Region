function [test_ty]=shengjingwangluo(data,L,N)
price=data'



%% 2.构造样本集
% 数据个数
n=length(price);

% 确保price为列向量
price=price(:);

% x(n) 由x(n-1),x(n-2),...,x(n-L)共L个数预测得到.


% price_n：每列为一个构造完毕的样本，共n-L个样本
price_n = zeros(L+1, n-L);
for i=1:n-L
    price_n(:,i) = price(i:i+L);
end

%% 划分训练、测试样本
% 将前280份数据划分为训练样本
% 后51份数据划分为测试样本

trainx = price_n(1:L-N+1,:);
trainy = price_n(L-N+2:end,:);
[ww,mm]=size(trainx);
testx = price(n-ww+1:n, end);


%% 创建Elman神经网络

% 包含15个神经元，训练函数为traingdx
net=elmannet(1:2,15,'traingdx');

% 设置显示级别
net.trainParam.show=1;

% 最大迭代次数为2000次
net.trainParam.epochs=3000;

% 误差容限，达到此误差就可以停止训练
net.trainParam.goal=0.000001;

% 最多验证失败次数
net.trainParam.max_fail=5;

% 对网络进行初始化
net=init(net);

%% 网络训练

%训练数据归一化
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);

% 测试数据做与训练数据相同的归一化操作
testx1 = mapminmax('apply',testx,st1);


% 输入训练样本进行训练
[net,per] = train(net,trainx1,trainy1);

%% 测试。输入归一化后的数据，再对实际输出进行反归一化

% 将训练数据输入网络进行测试
train_ty1 = sim(net, trainx1);
train_ty = mapminmax('reverse', train_ty1, st2);

% 将测试数据输入网络进行测试
test_ty1 = sim(net, testx1);
test_ty = mapminmax('reverse', test_ty1, st2);
  
%% 显示结果
% 1.显示训练数据的测试结果
figure(1)
x=1:length(train_ty);

% 显示真实值
plot(x,trainy,'b-');
hold on
% 显示神经网络的输出值
plot(x,train_ty,'r--')

legend('指标真实值','Elman网络输出值')
title('训练数据的测试结果');

% 显示残差
figure(2)
plot(x, train_ty - trainy)
title('训练数据测试结果的残差')

% 显示均方误差
mse1 = mse(train_ty - trainy);
fprintf('    mse = \n     %f\n', mse1)

% 显示相对误差
disp('    相对误差：')
fprintf('%f  ', (train_ty - trainy)./trainy );
fprintf('\n')
% 显示预测值
disp('    预测值：')
fprintf('%f  ', test_ty );
fprintf('\n')


