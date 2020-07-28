%% 初始化变量
clc;close all;clear all;warning off;%清除变量
format long g;
%% 读取数据
load BPoutput;
load JXoutput ;
load XBoutput;
load OUT;
%% 组合器决策
SH=[OUT(:,1),BPO(:,1),JXO(:,1),XBO(:,1)];
POR=[OUT(:,2),BPO(:,2),JXO(:,2),XBO(:,2)];
PERM=[OUT(:,3),BPO(:,3),JXO(:,3),XBO(:,3)];
Sg=[OUT(:,4),BPO(:,4),JXO(:,4),XBO(:,4)];
%泥质含量
for i=2:4
Wa=abs(SH(:,i)-SH(:,1))./(abs(SH(:,2)-SH(:,1)+abs(SH(:,3)-SH(:,1))+abs(SH(:,4)-SH(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CSH=SH(:,mn)*v1+SH(:,mx)*v2+SH(:,md)*v3;
%孔隙度
for i=2:4
Wa=abs(POR(:,i)-POR(:,1))./(abs(POR(:,2)-POR(:,1)+abs(POR(:,3)-POR(:,1))+abs(POR(:,4)-POR(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CPOR=POR(:,mn)*v1+POR(:,mx)*v2+POR(:,md)*v3;

% 渗透率
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
% 含气饱和度
for i=2:4
Wa=abs(Sg(:,i)-Sg(:,1))./(abs(Sg(:,2)-Sg(:,1)+abs(Sg(:,3)-Sg(:,1))+abs(Sg(:,4)-Sg(:,1))));
Wmean(:,i-1)=mean(Wa);
W1(:,i-1)=Wa;
end
[v1,mx]=max(Wmean);
[v2,mn]=min(Wmean);
v3=median(Wmean);[~,md]=find(Wmean==v3);

CSg=Sg(:,mn)*v1+Sg(:,mx)*v2+Sg(:,md)*v3;
CSg(CSg<0)=0;

%% 总结果
CM0=[CSH,CPOR,CPERM,CSg];
