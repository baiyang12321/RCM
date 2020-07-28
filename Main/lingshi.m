%% BP神经网络
clearvars -except hObject eventdata handles;
h = waitbar(0,'正在进行BP网络训练...');%%%%进度条
load inputdata2;

waitbar(0.1);%%%%

%%BP的构建,训练和测试
% 隐含层节点数选择
hidnumberset=1:2:40;
long1=length(hidnumberset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumberset(i),{'tansig','purelin'});%新建BP神经网络net purelin
    % 设置神经网络的参数
    net.trainparam.epochs=200;%训练次数
    net.trainparam.goal=0.00001;%精度
    net.trainparam.lr=0.1;%学习率
    net.trainParam.showWindow = false; % 不显示训练窗口
    [net,tr]=train(net,input_train,output_train);%训练
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
hidnumber=hidnumberset(index200);
waitbar(0.3);%%%%
% 学习率数选择
lrset=0.1:0.1:1;
long1=length(lrset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%新建BP神经网络net purelin
    % 设置神经网络的参数
    net.trainparam.epochs=200;%训练次数
    net.trainparam.goal=0.00001;%精度
    net.trainparam.lr=lrset(i);%学习率
    net.trainParam.showWindow = false; % 不显示训练窗口
    [net,tr]=train(net,input_train,output_train);%训练
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
lr=lrset(index200);
waitbar(0.5);%%%%
% 正式构建网络
net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%新建BP神经网络net
% 设置神经网络的参数
net.trainparam.epochs=200;%训练次数
net.trainparam.goal=0.00001;%精度
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

waitbar(0.7);%%%%

%%测试
% input_train0=input_train(:,1:size(Inputdata,2)-snumber);
% ybptrain=sim(net,input_train0);
% ybptrain=mapminmax('reverse',ybptrain,outputps);%预测数据反归一化

ybptest=sim(net,input_test);
ybptest=mapminmax('reverse',ybptest,outputps);%预测数据反归一化

waitbar(0.9);%%%%

BPO=ybptest';
OUT=T2';
save BPoutput BPO;
save OUT OUT;
save parameter1 hidnumber lr;
save netBP net;
bp_cnl=abs(BPO(:,1)-OUT(:,1))./OUT(:,1);
bp_cnl_average=mean(bp_cnl);
bp_rd=abs(BPO(:,2)-OUT(:,2))./OUT(:,2);
bp_rd_average=mean(bp_rd);
bp_rs=abs(BPO(:,3)-OUT(:,3))./OUT(:,3);
bp_rs_average=mean(bp_rs);

waitbar(1,h,'已完成');%%%%
pause(2);
close(h);
