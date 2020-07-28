function [MSE,R,RE,PredictOut] = ANFIS_test_r(ANFIS_rmodel,input_test, target_test)

PredictOut = evalfis(input_test', ANFIS_rmodel)';
sample_num = size(input_test,2);
SE = sum( (PredictOut-target_test).^2 );
MSE = SE/sample_num;
R = corrcoef(target_test, PredictOut);
RE=mean(abs(PredictOut-target_test)./target_test);

end