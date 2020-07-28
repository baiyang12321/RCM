%% BP神经网络
%% 清空环境变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 提取数据
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,12:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=300;%用于测试的样本个数（必改）
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% 定义训练集
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% 定义测试集
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);

%训练数据归一化
[input_train,inputps]=mapminmax(P1);%归一化默认为-1-1
[output_train,outputps]=mapminmax(T1);
%测试数据归一化
input_test=mapminmax('apply',P2,inputps);

%% BP的构建,训练和测试
% 隐含层节点数选择
hidnumberset=10:2:26;
long1=length(hidnumberset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumberset(i),{'tansig','purelin'});%新建BP神经网络net purelin
    % 设置神经网络的参数
    net.trainparam.epochs=200;%训练次数
    net.trainparam.goal=0.0000001;%精度
    net.trainparam.lr=0.1;%学习率
    net.trainParam.showWindow = false; % 不显示训练窗口
    [net,tr]=train(net,input_train,output_train);%训练
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
hidnumber=hidnumberset(index200);

% 学习率数选择
lrset=0.01:0.01:0.2;
long1=length(lrset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%新建BP神经网络net purelin
    % 设置神经网络的参数
    net.trainparam.epochs=200;%训练次数
    net.trainparam.goal=0.0000001;%精度
    net.trainparam.lr=lrset(i);%学习率
    net.trainParam.showWindow = false; % 不显示训练窗口
    [net,tr]=train(net,input_train,output_train);%训练
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
lr=lrset(index200);

% 正式构建网络
net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%新建BP神经网络net
% 设置神经网络的参数
net.trainparam.epochs=1000;%训练次数
net.trainparam.goal=0.0000001;%精度
net.trainparam.lr=lr;%学习率
net.trainParam.showWindow = false; % 不显示训练窗口
% net.trainFcn='trainlm';% LM算法
% net.trainFcn='traingd';% 梯度下降
% net.trainFcn='traingdm';% 有动量的梯度下降法
% net.trainFcn='traingda';% 自适应lr梯度下降法
% net.trainFcn='traingdx';% 自适应lr动量梯度下降法
net.trainFcn='trainrp';% 弹性梯度下降法
net.divideFcn ='';

[net,tr]=train(net,input_train,output_train);%训练
mse01=tr.perf;%训练误差
epochs=tr.epoch;% 训练次数


%% 测试
input_train0=input_train(:,1:size(Inputdata,2)-snumber);
ybptrain=sim(net,input_train0);
ybptrain=mapminmax('reverse',ybptrain,outputps);%预测数据反归一化

ybptest=sim(net,input_test);
ybptest=mapminmax('reverse',ybptest,outputps);%预测数据反归一化

BPO=ybptest';
OUT=T2';
save BPoutput BPO;
save OUT OUT;


%% 极限学习机回归拟合
%% 清空环境变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 导入数据
% 读取数据
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,12:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=300;%用于测试的样本个数
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% 定义训练集
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% 定义测试集
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);

% 训练数据归一化
[input_train,inputps]=mapminmax(P1);%归一化默认为-1-1
[output_train,outputps]=mapminmax(T1);
%测试数据归一化
input_test=mapminmax('apply',P2,inputps);

%% ELM创建/训练
emat=zeros(300,1);
for m=1:300;
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,m,'sig',0);

%% ELM仿真测试
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% 反归一化
T_sim = mapminmax('reverse',Tn_sim,outputps);

[R2,MSE,RMSE,MAPE,MAD]=predictorsfun(T2,T_sim);
emat(m,1)=R2;
end
[v1,N]=max(emat);
%% 正式构建极限学习机
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,N,'sig',0);

%% ELM仿真测试
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% 反归一化
T_sim = mapminmax('reverse',Tn_sim,outputps);


JXO=T_sim';
save JXoutput JXO;

%% 小波神经网络
%% 清空环境变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 提取数据
emat=zeros(300,2);
for style=12:13
% 读取数据
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,style));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=300;%用于测试的样本个数
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% 定义训练集
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% 定义测试集
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);
%% 网络参数配置

M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=15; %隐形节点个数
lr1=0.01; %学习概率
lr2=0.001; %学习概率
maxgen=1200; %迭代次数

%权值初始化
Wjk=randn(n,M);Wjk_1=Wjk;Wjk_2=Wjk_1;
Wij=randn(N,n);Wij_1=Wij;Wij_2=Wij_1;
a=randn(1,n);a_1=a;a_2=a_1;
b=randn(1,n);b_1=b;b_2=b_1;

%节点初始化
y=zeros(1,N);
net=zeros(1,n);
net_ab=zeros(1,n);

%权值学习增量初始化
d_Wjk=zeros(n,M);
d_Wij=zeros(N,n);
d_a=zeros(1,n);
d_b=zeros(1,n);

%% 输入输出数据归一化
[input_train,inputps]=mapminmax(P1);%归一化默认为-1-1
[output_train,outputps]=mapminmax(T1);
%测试数据归一化
input_test=mapminmax('apply',P2,inputps);

input_train=input_train';
output_train=output_train';
input_test=input_test';

error=zeros(1,maxgen);
%% 网络训练
for i=1:maxgen
    
    %误差累计
    error(i)=0;
    
    % 循环训练
    for kk=1:size(P1,1)
        x=input_train(kk,:);
        yqw=output_train(kk,:);
   
        for j=1:n
            for k=1:M
                net(j)=net(j)+Wjk(j,k)*x(k);
                net_ab(j)=(net(j)-b(j))/a(j);
            end
            temp=mymorlet(net_ab(j));
            for k=1:N
                y=y+Wij(k,j)*temp;   %小波函数
            end
        end
        
        %计算误差和
        error(i)=error(i)+sum(abs(yqw-y));
        
        %权值调整
        for j=1:n
            %计算d_Wij
            temp=mymorlet(net_ab(j));
            for k=1:N
                d_Wij(k,j)=d_Wij(k,j)-(yqw(k)-y(k))*temp;
            end
            %计算d_Wjk
            temp=d_mymorlet(net_ab(j));
            for k=1:M
                for l=1:N
                    d_Wjk(j,k)=d_Wjk(j,k)+(yqw(l)-y(l))*Wij(l,j) ;
                end
                d_Wjk(j,k)=-d_Wjk(j,k)*temp*x(k)/a(j);
            end
            %计算d_b
            for k=1:N
                d_b(j)=d_b(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_b(j)=d_b(j)*temp/a(j);
            %计算d_a
            for k=1:N
                d_a(j)=d_a(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_a(j)=d_a(j)*temp*((net(j)-b(j))/b(j))/a(j);
        end
        
        %权值参数更新      
        Wij=Wij-lr1*d_Wij;
        Wjk=Wjk-lr1*d_Wjk;
        b=b-lr2*d_b;
        a=a-lr2*d_a;
    
        d_Wjk=zeros(n,M);
        d_Wij=zeros(N,n);
        d_a=zeros(1,n);
        d_b=zeros(1,n);

        y=zeros(1,N);
        net=zeros(1,n);
        net_ab=zeros(1,n);
        
        Wjk_1=Wjk;Wjk_2=Wjk_1;
        Wij_1=Wij;Wij_2=Wij_1;
        a_1=a;a_2=a_1;
        b_1=b;b_2=b_1;
    end
end
%% 网络预测
yuce=zeros(numberTest,1);
%网络预测
for i=1:numberTest
    x_test=input_test(i,:);

    for j=1:1:n
        for k=1:1:M
            net(j)=net(j)+Wjk(j,k)*x_test(k);
            net_ab(j)=(net(j)-b(j))/a(j);
        end
        temp=mymorlet(net_ab(j));
        for k=1:N
            y(k)=y(k)+Wij(k,j)*temp ; 
        end
    end

    yuce(i)=y(k);
    y=zeros(1,N);
    net=zeros(1,n);
    net_ab=zeros(1,n);
end
%预测输出反归一化
ynn=mapminmax('reverse',yuce,outputps);
emat(:,style-11)=ynn;
end

XBO=emat;
save XBoutput XBO;
%% 委员会机器组合器
%% 初始化变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 读取数据
load BPoutput;
load JXoutput ;
load XBoutput;
load OUT;
%% 组合器决策
% SH=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
POR=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
PERM=[OUT(:,2),BPO(:,2),JXO(:,2),XBO(:,2)];
% Sg=[OUT(:,4),BPO(:,4),JXO(:,4),XBO(:,4)];
% %泥质含量
% for i=2:4
% Wa=abs(SH(:,i)-SH(:,1))./(abs(SH(:,2)-SH(:,1)+abs(SH(:,3)-SH(:,1))+abs(SH(:,4)-SH(:,1))));
% Wmean(:,i-1)=mean(Wa);
% W1(:,i-1)=Wa;
% end
% [v1,mx]=max(Wmean);
% [v2,mn]=min(Wmean);
% v3=median(Wmean);[~,md]=find(Wmean==v3);

% CSH=SH(:,mn)*v1+SH(:,mx)*v2+SH(:,md)*v3;
%孔隙度
for i=2:4
Wa=abs(POR(:,i)-POR(:,1))./(abs(POR(:,2)-POR(:,1)+abs(POR(:,3)-POR(:,1))+abs(POR(:,4)-POR(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CPOR=POR(:,mn)*v1+POR(:,mx)*v2+POR(:,md)*v3;

% 渗透率
for i=2:4
Wa=abs(PERM(:,i)-PERM(:,1))./(abs(PERM(:,2)-PERM(:,1)+abs(PERM(:,3)-PERM(:,1))+abs(PERM(:,4)-PERM(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CPERM=PERM(:,mn)*v1+PERM(:,mx)*v2+PERM(:,md)*v3;
CPERM(CPERM<0)=0;
% % 含气饱和度
% for i=2:4
% Wa=abs(Sg(:,i)-Sg(:,1))./(abs(Sg(:,2)-Sg(:,1)+abs(Sg(:,3)-Sg(:,1))+abs(Sg(:,4)-Sg(:,1))));
% Wmean(:,i-1)=mean(Wa);
% W1(:,i-1)=Wa;
% end
% [v1,mx]=max(Wmean);
% [v2,mn]=min(Wmean);
% v3=median(Wmean);[~,md]=find(Wmean==v3);
% 
% CSg=Sg(:,mn)*v1+Sg(:,mx)*v2+Sg(:,md)*v3;
% CSg(CSg<0)=0;

%% 总结果
CM0=[CPOR,CPERM];
e_por=abs(CPOR-POR(:,1))./POR(:,1);
e_por_average=mean(e_por);
e_perm=abs(CPERM-PERM(:,1))./PERM(:,1);
e_perm_average=mean(e_perm);