function [premat,vmat,accuracy]=fntnfun(typemat,typemat2)
% ���������ʺͼ�����
% ���� 
% typemat=ʵ�����
% typemat2=Ԥ�����

% ���
% premat=��������
% vmat=ÿ��Ԥ��׼ȷ��
% accuracy

accuracy=sum(typemat-typemat2==0)/length(typemat);% ׼ȷ��

set1=sort(unique(typemat));% ȫ������
n=length(set1);
premat=zeros(n,n);
vmat=zeros(n,1);
for i=1:n
    h1= typemat==set1(i);
    mat2= typemat2(h1);
    set2=sort(unique(mat2));
    n2=length(set2);
    for j=1:n2
        h2= mat2==set2(j);
        premat(set1(i),set2(j))=sum(h2);% ����set1(i)��Ԥ��Ϊset2(j)������
    end
    
    v201=sum(mat2==set1(i))/sum(h1);
    vmat(i,1)=v201;
end


