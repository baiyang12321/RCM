function indexM=S2indexMfun(S)

% �Ӹ�������ת��Ϊͳһ���±����
n=length(S);
indexM=zeros(n,2);
for i=1:n
    if i==1
        index1=1;
        index2=index1+S(i)-1;
    else
        index1=index2+1;
        index2=index1+S(i)-1;
    end
    indexM(i,1)=index1;%��ʼ�±�
    indexM(i,2)=index2;%�����±�
end