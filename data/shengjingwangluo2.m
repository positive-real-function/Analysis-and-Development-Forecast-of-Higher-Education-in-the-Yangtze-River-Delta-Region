
function [Y1]=shengjingwangluo2(A,B,testx)
x=A';
y=B';
[n1,m1]=size(A) 
[n2,m2]=size(B) 
n=m1  
m=m2  
 
p=x;  
t=y; 
[pn,minp,maxp,tn,mint,maxt]=premnmx(p,t); 
u=ones(n,1);
dx=[-1*u,1*u];                  
%BP网络训练
net=newff(dx,[n,15,m],{'tansig','tansig','purelin'},'traingdx'); 
 %建立模型，并用梯度下降法训练．
net.trainParam.show=1000;               
net.trainParam.Lr=0.05;                 
net.trainParam.epochs=5000;           
net.trainParam.goal=0.65*10^(-3);     
net=train(net,pn,tn);                   

an=sim(net,pn);          
a=postmnmx(an,mint,maxt);

pnew=testx;   
 
pnewn=tramnmx(pnew,minp,maxp); 
anewn=sim(net,pnewn);  
anew=postmnmx(anewn,mint,maxt) 
Y1=anew';







































