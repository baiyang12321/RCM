%% ����ѧϰ���ع����
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��������
% ��ȡ����
[adata,bdata,cdata]=xlsread('#1');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:9));
dataY=cell2mat(datacell(:,10:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=100;%���ڲ��Ե���������
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% ������Լ�
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);

% ѵ�����ݹ�һ��
[input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1-1
[output_train,outputps]=mapminmax(T1);
%�������ݹ�һ��
input_test=mapminmax('apply',P2,inputps);

%% ELM����/ѵ��
emat=zeros(20,1);
for m=1:20;
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,m,'sig',0);

%% ELM�������
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% ����һ��
T_sim = mapminmax('reverse',Tn_sim,outputps);

[R2,MSE,RMSE,MAPE,MAD]=predictorsfun(T2,T_sim);
emat(m,1)=R2;
end
[v1,N]=max(emat);
%% ��ʽ��������ѧϰ��
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,N,'sig',0);

%% ELM�������
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% ����һ��
T_sim = mapminmax('reverse',Tn_sim,outputps);


JXO=T_sim';
save JXoutput JXO;