function [MSE,R,RE,PredictOut] = ELM_test_r(ELM_rmodel,input_test, target_test)

PredictOut = elmpredict(input_test,ELM_rmodel{1},ELM_rmodel{2},ELM_rmodel{3},ELM_rmodel{4},ELM_rmodel{5});
sample_num = size(input_test,2);
SE = sum( (PredictOut-target_test).^2 );
MSE = SE/sample_num;
R = corrcoef(target_test, PredictOut);
RE=mean(abs(PredictOut-target_test)./target_test);

end