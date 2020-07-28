%% ������
clc;close all;clear all;warning off;%�������
format long g;

emat=zeros(10,4);
for style=10:13
% ��ȡ����
[adata,bdata,cdata]=xlsread('#1');
datacell=cdata(3:end,1:end);

dataX=cell2mat(datacell(:,5:9));
dataY=cell2mat(datacell(:,style));
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

Inputdata=dataX';
Outputdata=dataY';

snumber=size(Outputdata,2);%2�����еĸ���
index200=randperm(snumber);%�������
numberTest=10;%���ڲ��Ե���������
indextrain=index200;
indextest=index200(end-numberTest+1:end);
indextrain0=index200(1:end-numberTest);
% ����ѵ����
P1=Inputdata(:,1:125);
T1=Outputdata(:,1:125);
% ������Լ�
P2=Inputdata(:,126:135);
T2=Outputdata(:,126:135);
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
emat(:,style-9)=ynn;
end

XBO=emat;
save XBoutput XBO;