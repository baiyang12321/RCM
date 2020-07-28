%% 小波神经网络
%% 清空环境变量
clc;close all;clear all;warning off;%清除变量
format long g;
% 提取数据
load inputdata1 numberTest;
emat1=zeros(numberTest,2);

t=20;
emat=zeros(t,1);
h=waitbar(0,'please wait');
for m=1:t;
for style=12:13
%读取数据
[adata,bdata,cdata]=xlsread('#3');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:11));
dataY=cell2mat(datacell(:,12));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2代表列的个数
index200=randperm(snumber);%随机样本
numberTest=int16(snumber*0.2);%用于测试的样本个数
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
load inputdata1 indextrain;
% 定义训练集
P1=Inputdata(:,indextrain);
T1=Outputdata(:,indextrain);
% 定义测试集
P2=Inputdata(:,indextest);
T2=Outputdata(:,indextest);

%% 网络参数配置

M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=m; %隐形节点个数
lr1=0.001; %学习概率
lr2=0.0001; %学习概率
maxgen=200; %迭代次数

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
yuce=yuce';
ynn=mapminmax('reverse',yuce,outputps);
emat1(:,style-11)=ynn';
end

WNNO=emat1;
load inputdata1 T2;
OUT=T2';
save WNNoutput WNNO;

WNN_por=abs(WNNO(:,1)-OUT(:,1))./OUT(:,1);
WNN_por_average=mean(WNN_por);
WNN_perm=abs(WNNO(:,2)-OUT(:,2))./OUT(:,2);
WNN_perm_average=mean(WNN_perm);
emat(m,1)=WNN_por_average.*WNN_perm_average;
str=['参数优选中...',num2str(m/t*100),'%'];
waitbar(m/t,h,str);
end
delete(h);
[v1,nn]=min(emat);


%% 参数优化2
clearvars -EXCEPT nn
load inputdata1;
t=300;
emat=zeros(t,1);
h=waitbar(0,'please wait');
for ma=1:t
M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=nn; %隐形节点个数
lr1=0.01; %学习概率
lr2=0.001; %学习概率
maxgen=ma; %迭代次数

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
yuce=yuce';
ynn=mapminmax('reverse',yuce,outputps);
% emat(:,style-11)=ynn;
% end

WNNO=ynn';
OUT=T2';
save WNNoutput WNNO;

WNN_por=abs(WNNO(:,1)-OUT(:,1))./OUT(:,1);
WNN_por_average=mean(WNN_por);
WNN_perm=abs(WNNO(:,2)-OUT(:,2))./OUT(:,2);
WNN_perm_average=mean(WNN_perm);
emat(ma,1)=WNN_por_average.*WNN_perm_average;
str=['参数优选中...',num2str(ma/t*100),'%'];
waitbar(ma/t,h,str);
end
delete(h);
[v1,mm]=min(emat);

%% 参数优化3
clearvars -EXCEPT nn mm
load inputdata1;
emat=zeros(1000,1);
h=waitbar(0,'please wait');
for ll1=0.001:0.001:1
M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=nn; %隐形节点个数
lr1=ll1; %学习概率
lr2=0.001; %学习概率
maxgen=mm; %迭代次数

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
yuce=yuce';
ynn=mapminmax('reverse',yuce,outputps);
% emat(:,style-11)=ynn;
% end

WNNO=ynn';
OUT=T2';
save WNNoutput WNNO;

WNN_por=abs(WNNO(:,1)-OUT(:,1))./OUT(:,1);
WNN_por_average=mean(WNN_por);
WNN_perm=abs(WNNO(:,2)-OUT(:,2))./OUT(:,2);
WNN_perm_average=mean(WNN_perm);
ematll=int16(ll1*1000);
emat(ematll,1)=WNN_por_average.*WNN_perm_average;
str=['参数优选中...',num2str(ll1*100),'%'];
waitbar(ll1,h,str);
end
delete(h);
[v1,lll1]=min(emat);
lll1=lll1*0.001;


%% 参数优化4
clearvars -EXCEPT nn mm lll1;
load inputdata1;
emat=zeros(1000,1);
h=waitbar(0,'please wait');
for ll2=0.001:0.001:1
M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=nn; %隐形节点个数
lr1=lll1; %学习概率
lr2=ll2; %学习概率
maxgen=mm; %迭代次数

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
yuce=yuce';
ynn=mapminmax('reverse',yuce,outputps);
% emat(:,style-11)=ynn;
% end

WNNO=ynn';
OUT=T2';
save WNNoutput WNNO;

WNN_por=abs(WNNO(:,1)-OUT(:,1))./OUT(:,1);
WNN_por_average=mean(WNN_por);
WNN_perm=abs(WNNO(:,2)-OUT(:,2))./OUT(:,2);
WNN_perm_average=mean(WNN_perm);
ematll=int16(ll2*1000);
emat(ematll,1)=WNN_por_average.*WNN_perm_average;
str=['参数优选中...',num2str(ll2*100),'%'];
waitbar(ll2,h,str);
end
delete(h);
[v1,lll2]=min(emat);
lll2=lll2*0.001;

%% 正式构建
clearvars -EXCEPT nn mm lll1 lll2;
load inputdata1;


M=size(P1',2); %输入节点个数
N=size(T1',2); %输出节点个数

n=nn; %隐形节点个数
lr1=lll1; %学习概率
lr2=lll2; %学习概率
maxgen=mm; %迭代次数

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
yuce=yuce';
ynn=mapminmax('reverse',yuce,outputps);
% emat(:,style-11)=ynn;
% end

WNNO=ynn';
OUT=T2';
save WNNoutput WNNO;

WNN_por=abs(WNNO(:,1)-OUT(:,1))./OUT(:,1);
WNN_por_average=mean(WNN_por);
WNN_perm=abs(WNNO(:,2)-OUT(:,2))./OUT(:,2);
WNN_perm_average=mean(WNN_perm);