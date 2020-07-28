%% BP������
clc;close all;clear all;warning off;%�������
format long g;

%% ��ȡ����
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
numberTest=20;%���ڲ��Ե������������ظģ�

% ����ѵ����
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% ������Լ�
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);

%ѵ�����ݹ�һ��
[input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1-1
[output_train,outputps]=mapminmax(T1);
%�������ݹ�һ��
input_test=mapminmax('apply',P2,inputps);

%% BP�Ĺ���,ѵ���Ͳ���
% ������ڵ���ѡ��
hidnumberset=10:2:26;
long1=length(hidnumberset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumberset(i),{'tansig','purelin'});%�½�BP������net purelin
    % ����������Ĳ���
    net.trainparam.epochs=200;%ѵ������
    net.trainparam.goal=0.0000001;%����
    net.trainparam.lr=0.1;%ѧϰ��
    net.trainParam.showWindow = false; % ����ʾѵ������
    [net,tr]=train(net,input_train,output_train);%ѵ��
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
hidnumber=hidnumberset(index200);

% ѧϰ����ѡ��
lrset=0.01:0.01:0.2;
long1=length(lrset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%�½�BP������net purelin
    % ����������Ĳ���
    net.trainparam.epochs=200;%ѵ������
    net.trainparam.goal=0.0000001;%����
    net.trainparam.lr=lrset(i);%ѧϰ��
    net.trainParam.showWindow = false; % ����ʾѵ������
    [net,tr]=train(net,input_train,output_train);%ѵ��
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
lr=lrset(index200);

% ��ʽ��������
net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%�½�BP������net
% ����������Ĳ���
net.trainparam.epochs=1000;%ѵ������
net.trainparam.goal=0.0000001;%����
net.trainparam.lr=lr;%ѧϰ��
% net.trainFcn='trainlm';% LM�㷨
% net.trainFcn='traingd';% �ݶ��½�
% net.trainFcn='traingdm';% �ж������ݶ��½���
% net.trainFcn='traingda';% ����Ӧlr�ݶ��½���
% net.trainFcn='traingdx';% ����Ӧlr�����ݶ��½���
net.trainFcn='trainrp';% �����ݶ��½���
net.divideFcn ='';

[net,tr]=train(net,input_train,output_train);%ѵ��
mse01=tr.perf;%ѵ�����
epochs=tr.epoch;% ѵ������


%% ����
input_train0=input_train(:,1:size(Inputdata,2)-snumber);
ybptrain=sim(net,input_train0);
ybptrain=mapminmax('reverse',ybptrain,outputps);%Ԥ�����ݷ���һ��

ybptest=sim(net,input_test);
ybptest=mapminmax('reverse',ybptest,outputps);%Ԥ�����ݷ���һ��

BPO=ybptest';
OUT=T2';
save BPoutput BPO;
save OUT OUT;


