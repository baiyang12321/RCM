function ELM_rmodel = Elman_train_r(Input_train_P, Input_train_T, Elmanhidden_size)

net=newelm(minmax(Input_train_P),[Elmanhidden_size,1],{'tansig','tansig','tansig','purelin'},'traingdm','learngdm');
%%%     net = newelm(P,T,[S1...S(N-l)],{TF1...TFN},BTF,BLF,PF,IPF,OPF,DDF)
%%%     训练函数BTF可以是  最速下降BP算法:'traingd',  动量BP算法'traingdm',  学习率可变的最速下降BP算法'traingda',  学习率可变的动量BP算法'traingdx'
%%%     学习函数BLF可以是  'learngd',  'learngdm'
%%%     性能函数可以是  'mse','msereg'
% 设置网络训练参数

net.trainparam.epochs = 3000;
net.trainParam.lr = 0.02;
net.trainParam.mc = 0.9;
net.trainParam.goal = 1e-5;
net.trainparam.show = 50;
net.trainParam.showWindow = false; % 不显示训练窗口

ELM_rmodel=train(net,Input_train_P,Input_train_T);

end