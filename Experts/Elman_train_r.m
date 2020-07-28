function ELM_rmodel = Elman_train_r(Input_train_P, Input_train_T, Elmanhidden_size)

net=newelm(minmax(Input_train_P),[Elmanhidden_size,1],{'tansig','tansig','tansig','purelin'},'traingdm','learngdm');
%%%     net = newelm(P,T,[S1...S(N-l)],{TF1...TFN},BTF,BLF,PF,IPF,OPF,DDF)
%%%     ѵ������BTF������  �����½�BP�㷨:'traingd',  ����BP�㷨'traingdm',  ѧϰ�ʿɱ�������½�BP�㷨'traingda',  ѧϰ�ʿɱ�Ķ���BP�㷨'traingdx'
%%%     ѧϰ����BLF������  'learngd',  'learngdm'
%%%     ���ܺ���������  'mse','msereg'
% ��������ѵ������

net.trainparam.epochs = 3000;
net.trainParam.lr = 0.02;
net.trainParam.mc = 0.9;
net.trainParam.goal = 1e-5;
net.trainparam.show = 50;
net.trainParam.showWindow = false; % ����ʾѵ������

ELM_rmodel=train(net,Input_train_P,Input_train_T);

end