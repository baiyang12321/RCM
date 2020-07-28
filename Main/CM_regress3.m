%% BP������
%% ��ջ�������
clc;close all;clear all;warning off;%�������
% format long g;
%% ��ȡ����
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,[5 6 8]));
dataY=cell2mat(datacell(:,[7 9 10 11]));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=int16(snumber*0.2);%���ڲ��Ե������������ظģ�
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% ������Լ�
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);
save inputdata1;
%ѵ�����ݹ�һ��
[input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1~1
[output_train,outputps]=mapminmax(T1);
%�������ݹ�һ��
input_test=mapminmax('apply',P2,inputps);
save inputdata2;
%% BP�Ĺ���,ѵ���Ͳ���
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


%% ����
% input_train0=input_train(:,1:size(Inputdata,2)-snumber);
% ybptrain=sim(net,input_train0);
% ybptrain=mapminmax('reverse',ybptrain,outputps);%Ԥ�����ݷ���һ��

ybptest=sim(net,input_test);
ybptest=mapminmax('reverse',ybptest,outputps);%Ԥ�����ݷ���һ��

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
%% ����ѧϰ���ع����
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��������
load inputdata2;
% % ��ȡ����
% [adata,bdata,cdata]=xlsread('#3');
% datacell=cdata(3:end,1:end);
% 
% dataX=cell2mat(datacell(:,5:11));
% dataY=cell2mat(datacell(:,12:13));
% dataX(isnan(dataX))=0;
% dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��
% 
% Inputdata=dataX';
% Outputdata=dataY';
% 
% snumber=size(Outputdata,2);%2�����еĸ���
% index200=randperm(snumber);%�������
% numberTest=300;%���ڲ��Ե���������
% indextrain=index200;
% indextest=index200(end-numberTest+1:end);
% indextrain0=index200(1:end-numberTest);
% % ����ѵ����
% P1=Inputdata(:,indextrain);
% T1=Outputdata(:,indextrain);
% % ������Լ�
% P2=Inputdata(:,indextest);
% T2=Outputdata(:,indextest);
% 
% % ѵ�����ݹ�һ��
% [input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1-1
% [output_train,outputps]=mapminmax(T1);
% %�������ݹ�һ��
% input_test=mapminmax('apply',P2,inputps);

%% ELM����/ѵ��
t=50;
emat=zeros(t,1);
h=waitbar(0,'please wait');
for m=1:t;
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,m,'sig',0);

%% ELM�������
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% ����һ��
T_sim = mapminmax('reverse',Tn_sim,outputps);

[R2,MSE,RMSE,MAPE,MAD]=predictorsfun(T2,T_sim);
emat(m,1)=R2;
str=['������ѡ��...',num2str(m/t*100),'%'];
waitbar(m/t,h,str);
end
delete(h);
[v1,N]=max(emat);
%% ��ʽ��������ѧϰ��
[IW,B,LW,TF,TYPE] = elmtrain(input_train,output_train,N,'sig',0);

%% ELM�������
Tn_sim = elmpredict(input_test,IW,B,LW,TF,TYPE);
% ����һ��
T_sim = mapminmax('reverse',Tn_sim,outputps);


ELMO=T_sim';
OUT=T2';
save JXoutput ELMO;
save parameter2 N;
save elmpredict IW B LW TF TYPE;
elm_cnl=abs(ELMO(:,1)-OUT(:,1))./OUT(:,1);
elm_cnl_average=mean(elm_cnl);
elm_rd=abs(ELMO(:,2)-OUT(:,2))./OUT(:,2);
elm_rd_average=mean(elm_rd);
elm_rs=abs(ELMO(:,3)-OUT(:,3))./OUT(:,3);
elm_rs_average=mean(elm_rs);
%% GRNN������
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��ȡ����
load inputdata2;
P1=P1';
T1=T1';
%% ������֤
desired_spread=[];
mse_max=10e20;
desired_input=[];
desired_output=[];
result_perfp=[];
indices = crossvalind('Kfold',length(P1),4);
h=waitbar(0,'����Ѱ�����Ż�����....');
k=1;
for i = 1:4
    perfp=[];
%     disp(['����Ϊ��',num2str(i),'�ν�����֤���'])
    test = (indices == i); train = ~test;
    p_cv_train=P1(train,:);
    t_cv_train=T1(train,:);
    p_cv_test=P1(test,:);
    t_cv_test=T1(test,:);
    p_cv_train=p_cv_train';
    t_cv_train=t_cv_train';
    p_cv_test= p_cv_test';
    t_cv_test= t_cv_test';
    [p_cv_train,minp,maxp,t_cv_train,mint,maxt]=premnmx(p_cv_train,t_cv_train);
    p_cv_test=tramnmx(p_cv_test,minp,maxp);
    for spread=0.01:0.01:1;
        net=newgrnn(p_cv_train,t_cv_train,spread);
%         disp(['��ǰspreadֵΪ', num2str(spread)]);
        test_Out=sim(net,p_cv_test);
        test_Out=postmnmx(test_Out,mint,maxt);
        error=t_cv_test-test_Out;
%         disp(['��ǰ�����mseΪ',num2str(mse(error))])
        perfp=[perfp mse(error)];
        if mse(error)<mse_max
            mse_max=mse(error);
            desired_spread=spread;
            desired_input=p_cv_train;
            desired_output=t_cv_train;
        end
        k=k+1;
     waitbar(k/400,h);
    end
    result_perfp(i,:)=perfp;
end;
close(h)
% disp(['���spreadֵΪ',num2str(desired_spread)])
% disp(['��ʱ�������ֵΪ'])
% desired_input;
% disp(['��ʱ������ֵΪ'])
% desired_output;


%% ������ѷ�������GRNN����
net=newgrnn(desired_input,desired_output,desired_spread);
P2=tramnmx(P2,minp,maxp);
grnn_prediction_result=sim(net,P2);
grnn_prediction_result=postmnmx(grnn_prediction_result,mint,maxt);
%% �������
GRNNO=grnn_prediction_result';
OUT=T2';
GRNN_cnl=abs(GRNNO(:,1)-OUT(:,1))./OUT(:,1);
GRNN_cnl_average=mean(GRNN_cnl);
GRNN_rd=abs(GRNNO(:,2)-OUT(:,2))./OUT(:,2);
GRNN_rd_average=mean(GRNN_rd);
GRNN_rs=abs(GRNNO(:,2)-OUT(:,2))./OUT(:,2);
GRNN_rs_average=mean(GRNN_rs);
save GRNNoutput GRNNO;
save parameter3 desired_spread minp maxp mint maxt;
save netGRNN net;
%% ίԱ����������
%% ��ʼ������
clc;close all;clear all;warning off;%�������
format long g;
%% ��ȡ����
load BPoutput;
load JXoutput ;
load GRNNoutput;
load OUT;
%% ���������
% SH=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
DEN=[OUT(:,1),BPO(:,1),ELMO(:,1),GRNNO(:,1)];
CNL=[OUT(:,2),BPO(:,2),ELMO(:,2),GRNNO(:,2)];
RD=[OUT(:,3),BPO(:,3),ELMO(:,3),GRNNO(:,3)];
RS=[OUT(:,4),BPO(:,4),ELMO(:,4),GRNNO(:,4)];

% %���ʺ���
% for i=2:4
% Wa=abs(SH(:,i)-SH(:,1))./(abs(SH(:,2)-SH(:,1)+abs(SH(:,3)-SH(:,1))+abs(SH(:,4)-SH(:,1))));
% Wmean(:,i-1)=mean(Wa);
% W1(:,i-1)=Wa;
% end
% [v1,mx]=max(Wmean);
% [v2,mn]=min(Wmean);
% v3=median(Wmean);[~,md]=find(Wmean==v3);

% CSH=SH(:,mn)*v1+SH(:,mx)*v2+SH(:,md)*v3;
%�ܶ�
for i=2:4
Wa=abs(DEN(:,i)-DEN(:,1))./(abs(DEN(:,2)-DEN(:,1))+abs(DEN(:,3)-DEN(:,1))+abs(DEN(:,4)-DEN(:,1)));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CDEN=DEN(:,mn)*v1+DEN(:,mx)*v2+DEN(:,md)*v3;
save CM_DEN_parameter v1 v2 v3 mn mx md; 
%���ӿ�϶
for i=2:4
Wa=abs(CNL(:,i)-CNL(:,1))./(abs(CNL(:,2)-CNL(:,1))+abs(CNL(:,3)-CNL(:,1))+abs(CNL(:,4)-CNL(:,1)));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CCNL=CNL(:,mn)*v1+CNL(:,mx)*v2+CNL(:,md)*v3;
save CM_CNL_parameter v1 v2 v3 mn mx md; 
% RD
for i=2:4
Wa=abs(RD(:,i)-RD(:,1))./(abs(RD(:,2)-RD(:,1))+abs(RD(:,3)-RD(:,1))+abs(RD(:,4)-RD(:,1)));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CRD=RD(:,mn)*v1+RD(:,mx)*v2+RD(:,md)*v3;
CRD(CRD<0)=0;
save CM_RD_parameter v1 v2 v3 mn mx md; 

% RS
for i=2:4
Wa=abs(RS(:,i)-RS(:,1))./(abs(RS(:,2)-RS(:,1))+abs(RS(:,3)-RS(:,1))+abs(RS(:,4)-RS(:,1)));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CRS=RS(:,mn)*v1+RS(:,mx)*v2+RS(:,md)*v3;
CRS(CRS<0)=0;
save CM_RS_parameter v1 v2 v3 mn mx md; 
% % �������Ͷ�
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

%% �ܽ��
CM0=[CDEN,CCNL,CRD,CRS];
e_den=abs(CDEN-DEN(:,1))./DEN(:,1);
e_den_average=mean(e_den);
e_cnl=abs(CCNL-CNL(:,1))./CNL(:,1);
e_cnl_average=mean(e_cnl);
e_rd=abs(CRD-RD(:,1))./RD(:,1);
e_rd_average=mean(e_rd);
e_rs=abs(CRS-RS(:,1))./RS(:,1);
e_rs_average=mean(e_rs);