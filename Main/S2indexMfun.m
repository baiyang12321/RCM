function indexM=S2indexMfun(S)

% 从个数矩阵转换为统一的下标矩阵
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
    indexM(i,1)=index1;%起始下标
    indexM(i,2)=index2;%结束下标
end