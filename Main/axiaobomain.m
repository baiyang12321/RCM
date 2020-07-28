%% 神经网络
clc;close all;clear all;warning off;%清除变量
format long g;

emat=zeros(10,4);
for style=10:13
% 读取数据
[adata,bdata,cdata]=xlsread('#1');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:9));
dataY=cell2mat(datacell(:,style));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=10;%用于测试的样本个数
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% 定义训练集
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% 定义测试集
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);
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
emat(:,style-9)=ynn;
end

XBO=emat;
save XBoutput XBO;