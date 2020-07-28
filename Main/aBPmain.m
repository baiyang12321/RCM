%% BP神经网络
clc;close all;clear all;warning off;%清除变量
format long g;

%% 提取数据
[adata,bdata,cdata]=xlsread('#1');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:9));
dataY=cell2mat(datacell(:,10:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=20;%用于测试的样本个数（必改）

% 定义训练集
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% 定义测试集
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);

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


