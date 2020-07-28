function [BPNN_model] = BPNN_train_r(input_train, target_train, hidden_size)
% 训练神经网络，最终得到回归BPNN模型
%
% BPNN_train(input_train, label_train, hidden_size, class_num)
%   input_train: 用于训练的输入特征矩阵, number_of_features x number_of_samples
%   target_train: 用于训练的目标值向量, 1 x number_of_samples
%   hidden_size: BPNN的隐含层神经元数量
%
% Example:
%   BPNN_train([  0.1882 -0.3653; -0.5107 -0.7358; 0.7536  0.8212;
%              0.0293 -0.3660; -0.8185 -0.7370; -0.8053 -0.8061; -0.9298 -0.9137],
%              [4 1], 10, 5)
%   多隐含层表示为：net=newff(input_train, label_train, [5,5])
% ================================================================
% net = newff(input_train,target_train,hidden_size,{'tansig','purelin'},'traingdm');
net = newff(input_train,target_train,hidden_size);
%传递函数TF: 　　purelin： 线性传递函数
%传递函数TF:  　　tansig ：正切S型传递函数
%传递函数TF:  　　logsig ：对数S型传递函数
%　　
% 学习训练函数BTF:   　traingd：最速下降BP算法。
% 学习训练函数BTF:   　traingdm：动量BP算法。
% 学习训练函数BTF:   　trainda：学习率可变的最速下降BP算法。
% 学习训练函数BTF:   　traindx：学习率可变的动量BP算法。
% 学习训练函数BTF:  　 trainrp：弹性算法。
% 学习训练函数BTF:  　　变梯度算法：
% 学习训练函数BTF:   　traincgf（Fletcher-Reeves修正算法）
% 学习训练函数BTF:   　traincgp（Polak_Ribiere修正算法）
% 学习训练函数BTF:   　traincgb（Powell-Beale复位算法）
% 学习训练函数BTF:   　trainbfg（BFGS 拟牛顿算法）
% 学习训练函数BTF:   　trainoss（OSS算法）

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.divideFcn='';
net.trainParam.showWindow = false;
BPNN_model = train(net,input_train,target_train);
end