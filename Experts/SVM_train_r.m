function SVM_rmodel = SVM_train_r(input_train, target_train)

% SVM_rmodel = fitrsvm(input_train', target_train', 'BoxConstraint',g_SVM_C,'KernelFunction','gaussian','KernelScale',g_SVM_gamma);
% svr = fitrsvm(X,Y,'Standardize',true,'epsilon',0.3,'kernelfunction','gaussian','KernelScale','auto');
SVM_rmodel = fitrsvm(input_train', target_train','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
%%%使用来自动优化超参数fitrsvm
end