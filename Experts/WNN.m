%% ��ջ�������

clc;
clear all
close all
nntwarn off;

%% ��������
load data2.mat;
input=Inputdata;
output=Outputdata;
% �������ѵ���������Լ�
% k = randperm(size(input,1));
k=[29,31,7,13,27,19,30,11,14,24,3,25,8,4,5,22,18,1,2,10,16,17,9,26,23,21,15,12,20,28,6];
N = 20;
% ѵ��������25������
P_train=input(k(1:N),:)';
T_train=output(k(1:N))';
% ���Լ�����6������
P_test=input(k(N+1:end),:)';
T_test=output(k(N+1:end))';

% %% ��һ��
% % ѵ����
% [Pn_train,inputps] = mapminmax(P_train);
% [Tn_train,outputps] = mapminmax(T_train);
% 
% % ���Լ�
% Pn_test = mapminmax('apply',P_test,inputps);

%% �����������
% load traffic_flux input output input_test output_test

input=P_train';
output=T_train';
input_test=P_test';

M=size(input,2); %����ڵ����
N=size(output,2); %����ڵ����

n=22; %���νڵ����
lr1=0.01; %ѧϰ����
lr2=0.001; %ѧϰ����
maxgen=100; %��������

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
[inputn,inputps]=mapminmax(input');
[outputn,outputps]=mapminmax(output'); 
inputn=inputn';
outputn=outputn';

error=zeros(1,maxgen);
%% ����ѵ��
for i=1:maxgen
    
    %����ۼ�
    error(i)=0;
    
    % ѭ��ѵ��
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
%Ԥ�������һ��
x=mapminmax('apply',input_test',inputps);
x=x';
yuce=zeros(size(x,1),1);
%����Ԥ��
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
%Ԥ���������һ��
test_out=mapminmax('reverse',yuce,outputps);
Re_CM=mean(abs(test_out'-T_test)./T_test)