function [test_ty]=shengjingwangluo(data,L,N)
price=data'



%% 2.����������
% ���ݸ���
n=length(price);

% ȷ��priceΪ������
price=price(:);

% x(n) ��x(n-1),x(n-2),...,x(n-L)��L����Ԥ��õ�.


% price_n��ÿ��Ϊһ��������ϵ���������n-L������
price_n = zeros(L+1, n-L);
for i=1:n-L
    price_n(:,i) = price(i:i+L);
end

%% ����ѵ������������
% ��ǰ280�����ݻ���Ϊѵ������
% ��51�����ݻ���Ϊ��������

trainx = price_n(1:L-N+1,:);
trainy = price_n(L-N+2:end,:);
[ww,mm]=size(trainx);
testx = price(n-ww+1:n, end);


%% ����Elman������

% ����15����Ԫ��ѵ������Ϊtraingdx
net=elmannet(1:2,15,'traingdx');

% ������ʾ����
net.trainParam.show=1;

% ����������Ϊ2000��
net.trainParam.epochs=3000;

% ������ޣ��ﵽ�����Ϳ���ֹͣѵ��
net.trainParam.goal=0.000001;

% �����֤ʧ�ܴ���
net.trainParam.max_fail=5;

% ��������г�ʼ��
net=init(net);

%% ����ѵ��

%ѵ�����ݹ�һ��
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);

% ������������ѵ��������ͬ�Ĺ�һ������
testx1 = mapminmax('apply',testx,st1);


% ����ѵ����������ѵ��
[net,per] = train(net,trainx1,trainy1);

%% ���ԡ������һ��������ݣ��ٶ�ʵ��������з���һ��

% ��ѵ����������������в���
train_ty1 = sim(net, trainx1);
train_ty = mapminmax('reverse', train_ty1, st2);

% ��������������������в���
test_ty1 = sim(net, testx1);
test_ty = mapminmax('reverse', test_ty1, st2);
  
%% ��ʾ���
% 1.��ʾѵ�����ݵĲ��Խ��
figure(1)
x=1:length(train_ty);

% ��ʾ��ʵֵ
plot(x,trainy,'b-');
hold on
% ��ʾ����������ֵ
plot(x,train_ty,'r--')

legend('ָ����ʵֵ','Elman�������ֵ')
title('ѵ�����ݵĲ��Խ��');

% ��ʾ�в�
figure(2)
plot(x, train_ty - trainy)
title('ѵ�����ݲ��Խ���Ĳв�')

% ��ʾ�������
mse1 = mse(train_ty - trainy);
fprintf('    mse = \n     %f\n', mse1)

% ��ʾ������
disp('    �����')
fprintf('%f  ', (train_ty - trainy)./trainy );
fprintf('\n')
% ��ʾԤ��ֵ
disp('    Ԥ��ֵ��')
fprintf('%f  ', test_ty );
fprintf('\n')


