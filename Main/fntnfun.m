function [premat,vmat,accuracy]=fntnfun(typemat,typemat2)
% 计算真正率和假正率
% 输入 
% typemat=实际类别
% typemat2=预测类别

% 输出
% premat=混淆矩阵
% vmat=每类预测准确率
% accuracy

accuracy=sum(typemat-typemat2==0)/length(typemat);% 准确率

set1=sort(unique(typemat));% 全部类型
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
        premat(set1(i),set2(j))=sum(h2);% 类型set1(i)被预测为set2(j)的数量
    end
    
    v201=sum(mat2==set1(i))/sum(h1);
    vmat(i,1)=v201;
end


