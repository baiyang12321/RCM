function [MSE,R,RE,PredictOut] = Elman_test_r(Elman_rmodel, input_test, target_test)
% BPNN_model: ֮ǰ�Ѿ�ѵ���õ�BPNNģ��
% input_test: ���ڲ��Եľ���number_of_samples-number_of_features
% target_test: ���ڲ��Ե�Ŀ�꣬number_of_samples-1

PredictOut = sim(Elman_rmodel, input_test);    %����BPNN�õ�Ԥ����

sample_num = size(input_test,2);
SE = sum( (PredictOut-target_test).^2 );
MSE = SE/sample_num;
R = corrcoef(target_test, PredictOut);
RE=mean(abs(PredictOut-target_test)./target_test);
end