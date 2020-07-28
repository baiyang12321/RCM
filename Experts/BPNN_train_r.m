function [BPNN_model] = BPNN_train_r(input_train, target_train, hidden_size)
% ѵ�������磬���յõ��ع�BPNNģ��
%
% BPNN_train(input_train, label_train, hidden_size, class_num)
%   input_train: ����ѵ����������������, number_of_features x number_of_samples
%   target_train: ����ѵ����Ŀ��ֵ����, 1 x number_of_samples
%   hidden_size: BPNN����������Ԫ����
%
% Example:
%   BPNN_train([  0.1882 -0.3653; -0.5107 -0.7358; 0.7536  0.8212;
%              0.0293 -0.3660; -0.8185 -0.7370; -0.8053 -0.8061; -0.9298 -0.9137],
%              [4 1], 10, 5)
%   ���������ʾΪ��net=newff(input_train, label_train, [5,5])
% ================================================================
% net = newff(input_train,target_train,hidden_size,{'tansig','purelin'},'traingdm');
net = newff(input_train,target_train,hidden_size);
%���ݺ���TF: ����purelin�� ���Դ��ݺ���
%���ݺ���TF:  ����tansig ������S�ʹ��ݺ���
%���ݺ���TF:  ����logsig ������S�ʹ��ݺ���
%����
% ѧϰѵ������BTF:   ��traingd�������½�BP�㷨��
% ѧϰѵ������BTF:   ��traingdm������BP�㷨��
% ѧϰѵ������BTF:   ��trainda��ѧϰ�ʿɱ�������½�BP�㷨��
% ѧϰѵ������BTF:   ��traindx��ѧϰ�ʿɱ�Ķ���BP�㷨��
% ѧϰѵ������BTF:  �� trainrp�������㷨��
% ѧϰѵ������BTF:  �������ݶ��㷨��
% ѧϰѵ������BTF:   ��traincgf��Fletcher-Reeves�����㷨��
% ѧϰѵ������BTF:   ��traincgp��Polak_Ribiere�����㷨��
% ѧϰѵ������BTF:   ��traincgb��Powell-Beale��λ�㷨��
% ѧϰѵ������BTF:   ��trainbfg��BFGS ��ţ���㷨��
% ѧϰѵ������BTF:   ��trainoss��OSS�㷨��

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.divideFcn='';
net.trainParam.showWindow = false;
BPNN_model = train(net,input_train,target_train);
end