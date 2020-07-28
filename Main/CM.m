%% BP������
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��ȡ����
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,12:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=300;%���ڲ��Ե������������ظģ�
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% ������Լ�
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);

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
net.trainParam.showWindow = false; % ����ʾѵ������
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


%% ����ѧϰ���ع����
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��������
% ��ȡ����
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,12:13));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=300;%���ڲ��Ե���������
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% ������Լ�
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);

% ѵ�����ݹ�һ��
[input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1-1
[output_train,outputps]=mapminmax(T1);
%�������ݹ�һ��
input_test=mapminmax('apply',P2,inputps);

%% ELM����/ѵ��
emat=zeros(300,1);
for m=1:300;
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

%% С��������
%% ��ջ�������
clc;close all;clear all;warning off;%�������
format long g;
%% ��ȡ����
emat=zeros(300,2);
for style=12:13
% ��ȡ����
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,style));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=300;%���ڲ��Ե���������
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% ������Լ�
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);
%% �����������

M=size(P1',2); %����ڵ����
N=size(T1',2); %����ڵ����

n=15; %���νڵ����
lr1=0.01; %ѧϰ����
lr2=0.001; %ѧϰ����
maxgen=1200; %��������

%Ȩֵ��ʼ��
Wjk=randn(n,M);Wjk_1=Wjk;Wjk_2=Wjk_1;
Wij=randn(N,n);Wij_1=Wij;Wij_2=Wij_1;
a=randn(1,n);a_1=a;a_2=a_1;
b=randn(1,n);b_1=b;b_2=b_1;

%�ڵ��ʼ��
y=zeros(1,N);
net=zeros(1,n);
net_ab=zeros(1,n);

%Ȩֵѧϰ������ʼ��
d_Wjk=zeros(n,M);
d_Wij=zeros(N,n);
d_a=zeros(1,n);
d_b=zeros(1,n);

%% ����������ݹ�һ��
[input_train,inputps]=mapminmax(P1);%��һ��Ĭ��Ϊ-1-1
[output_train,outputps]=mapminmax(T1);
%�������ݹ�һ��
input_test=mapminmax('apply',P2,inputps);

input_train=input_train';
output_train=output_train';
input_test=input_test';

error=zeros(1,maxgen);
%% ����ѵ��
for i=1:maxgen
    
    %����ۼ�
    error(i)=0;
    
    % ѭ��ѵ��
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
                y=y+Wij(k,j)*temp;   %С������
            end
        end
        
        %��������
        error(i)=error(i)+sum(abs(yqw-y));
        
        %Ȩֵ����
        for j=1:n
            %����d_Wij
            temp=mymorlet(net_ab(j));
            for k=1:N
                d_Wij(k,j)=d_Wij(k,j)-(yqw(k)-y(k))*temp;
            end
            %����d_Wjk
            temp=d_mymorlet(net_ab(j));
            for k=1:M
                for l=1:N
                    d_Wjk(j,k)=d_Wjk(j,k)+(yqw(l)-y(l))*Wij(l,j) ;
                end
                d_Wjk(j,k)=-d_Wjk(j,k)*temp*x(k)/a(j);
            end
            %����d_b
            for k=1:N
                d_b(j)=d_b(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_b(j)=d_b(j)*temp/a(j);
            %����d_a
            for k=1:N
                d_a(j)=d_a(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_a(j)=d_a(j)*temp*((net(j)-b(j))/b(j))/a(j);
        end
        
        %Ȩֵ��������      
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
%% ����Ԥ��
yuce=zeros(numberTest,1);
%����Ԥ��
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
%Ԥ���������һ��
ynn=mapminmax('reverse',yuce,outputps);
emat(:,style-11)=ynn;
end

XBO=emat;
save XBoutput XBO;
%% ίԱ����������
%% ��ʼ������
clc;close all;clear all;warning off;%�������
format long g;
%% ��ȡ����
load BPoutput;
load JXoutput ;
load XBoutput;
load OUT;
%% ���������
% SH=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
POR=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
PERM=[OUT(:,2),BPO(:,2),JXO(:,2),XBO(:,2)];
% Sg=[OUT(:,4),BPO(:,4),JXO(:,4),XBO(:,4)];
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
%��϶��
for i=2:4
Wa=abs(POR(:,i)-POR(:,1))./(abs(POR(:,2)-POR(:,1)+abs(POR(:,3)-POR(:,1))+abs(POR(:,4)-POR(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CPOR=POR(:,mn)*v1+POR(:,mx)*v2+POR(:,md)*v3;

% ��͸��
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
CM0=[CPOR,CPERM];
e_por=abs(CPOR-POR(:,1))./POR(:,1);
e_por_average=mean(e_por);
e_perm=abs(CPERM-PERM(:,1))./PERM(:,1);
e_perm_average=mean(e_perm);