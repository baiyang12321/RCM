function [MSE,R,RE,PredictOut] = RBF_test_r(ANFIS_rmodel,input_test, target_test)

PredictOut = sim(ANFIS_rmodel , input_test);
sample_num = size(input_test,2);
SE = sum( (PredictOut-target_test).^2 );
MSE = SE/sample_num;
R = corrcoef(target_test, PredictOut);
RE=mean(abs(PredictOut-target_test)./target_test);

end