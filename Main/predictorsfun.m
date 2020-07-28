function [R2,MSE,RMSE,MAPE,MAD]=predictorsfun(y,y1)


%% �������Ԥ�����ݵ�ָ��,�ɾ�ϵ��,���ƽ���͵�
% ����ɾ�ϵ��,mse��
% R2=1-sum((y-y1).^2)/sum((y-mean(y)).^2);%�ɾ�ϵ��
Q=sum((y-y1).^2);
R2=1-sqrt(Q/sum(y.^2));% ����Ŷ�
MSE=sum((y-y1).^2)/(length(y)-1);% �������
RMSE=sqrt(MSE);% ���������
MAPE=sum(abs(y1-y)*100./y)/length(y);% ��԰ٷ����
MAD=mean(abs(y1-y));% ƽ���������

