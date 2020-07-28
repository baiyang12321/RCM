%% 极限学习机回归拟合
%% 清空环境变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 导入数据
% 读取数据
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
numberTest=100;%用于测试的样本个数
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% 定义训练集
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% 定义测试集
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);

% 训练数据归一化
[input_train,inputps]=mapminmax(P1);%归一化默认为-1-1
[output_train,outputps]=mapminmax(T1);
%测试数据归一化
input_test=mapminmax('apply',P2,inputps);

%% ELM创建/训练
emat=zeros(20,1);
for m=1:20;
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