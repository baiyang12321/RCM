%% BP������
clearvars -except hObject eventdata handles;
h = waitbar(0,'���ڽ���BP����ѵ��...');%%%%������
load inputdata2;

waitbar(0.1);%%%%

%%BP�Ĺ���,ѵ���Ͳ���
% ������ڵ���ѡ��
hidnumberset=1:2:40;
long1=length(hidnumberset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumberset(i),{'tansig','purelin'});%�½�BP������net purelin
    % ����������Ĳ���
    net.trainparam.epochs=200;%ѵ������
    net.trainparam.goal=0.00001;%����
    net.trainparam.lr=0.1;%ѧϰ��
    net.trainParam.showWindow = false; % ����ʾѵ������
    [net,tr]=train(net,input_train,output_train);%ѵ��
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
hidnumber=hidnumberset(index200);
waitbar(0.3);%%%%
% ѧϰ����ѡ��
lrset=0.1:0.1:1;
long1=length(lrset);
emat1=zeros(long1,1);
for i=1:long1
    net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%�½�BP������net purelin
    % ����������Ĳ���
    net.trainparam.epochs=200;%ѵ������
    net.trainparam.goal=0.00001;%����
    net.trainparam.lr=lrset(i);%ѧϰ��
    net.trainParam.showWindow = false; % ����ʾѵ������
    [net,tr]=train(net,input_train,output_train);%ѵ��
    mse01=tr.perf;
    emat1(i,1)=mse01(end);
end
[v1,index200]=min(emat1);
lr=lrset(index200);
waitbar(0.5);%%%%
% ��ʽ��������
net=newff(input_train,output_train,hidnumber,{'tansig','purelin'});%�½�BP������net
% ����������Ĳ���
net.trainparam.epochs=200;%ѵ������
net.trainparam.goal=0.00001;%����
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

waitbar(0.7);%%%%

%%����
% input_train0=input_train(:,1:size(Inputdata,2)-snumber);
% ybptrain=sim(net,input_train0);
% ybptrain=mapminmax('reverse',ybptrain,outputps);%Ԥ�����ݷ���һ��

ybptest=sim(net,input_test);
ybptest=mapminmax('reverse',ybptest,outputps);%Ԥ�����ݷ���һ��

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

waitbar(1,h,'�����');%%%%
pause(2);
close(h);
