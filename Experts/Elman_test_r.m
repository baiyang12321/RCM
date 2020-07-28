function [MSE,R,RE,PredictOut] = Elman_test_r(Elman_rmodel, input_test, target_test)
% BPNN_model: 之前已经训练好的BPNN模型
% input_test: 用于测试的矩阵，number_of_samples-number_of_features
% target_test: 用于测试的目标，number_of_samples-1

PredictOut = sim(Elman_rmodel, input_test);    %输入BPNN得到预测结果

sample_num = size(input_test,2);
SE = sum( (PredictOut-target_test).^2 );
MSE = SE/sample_num;
R = corrcoef(target_test, PredictOut);
RE=mean(abs(PredictOut-target_test)./target_test);
end