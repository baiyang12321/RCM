%% 清空环境变量

clc;
clear all
close all
nntwarn off;

%% 导入数据
load data2.mat;
input=Inputdata;
output=Outputdata;
% 随机生成训练集、测试集
% k = randperm(size(input,1));
k=[29,31,7,13,27,19,30,11,14,24,3,25,8,4,5,22,18,1,2,10,16,17,9,26,23,21,15,12,20,28,6];
N = 20;
% 训练集——25个样本
P_train=input(k(1:N),:)';
T_train=output(k(1:N))';
% 测试集——6个样本
P_test=input(k(N+1:end),:)';
T_test=output(k(N+1:end))';

% %% 归一化
% % 训练集
% [Pn_train,inputps] = mapminmax(P_train);
% [Tn_train,outputps] = mapminmax(T_train);
% 
% % 测试集
% Pn_test = mapminmax('apply',P_test,inputps);

%% 网络参数配置
% load traffic_flux input output input_test output_test

input=P_train';
output=T_train';
input_test=P_test';

M=size(input,2); %输入节点个数
N=size(output,2); %输出节点个数

n=22; %隐形节点个数
lr1=0.01; %学习概率
lr2=0.001; %学习概率
maxgen=100; %迭代次数

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
[inputn,inputps]=mapminmax(input');
[outputn,outputps]=mapminmax(output'); 
inputn=inputn';
outputn=outputn';

error=zeros(1,maxgen);
%% 网络训练
for i=1:maxgen
    
    %误差累计
    error(i)=0;
    
    % 循环训练
    for kk=1:size(input,1)
        x=inputn(kk,:);
        yqw=outputn(kk,:);
   
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
%预测输入归一化
x=mapminmax('apply',input_test',inputps);
x=x';
yuce=zeros(size(x,1),1);
%网络预测
for i=1:size(x,1)
    x_test=x(i,:);

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
test_out=mapminmax('reverse',yuce,outputps);
Re_CM=mean(abs(test_out'-T_test)./T_test)