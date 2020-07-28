function [ANFIS_rmodel] = ANFIS_train_r(input_train, target_train, g_NF_clusterNum)
Input_train0=[input_train; target_train]';
opt = NaN(4,1);
opt(4) = 0;
in_fis = genfis3(input_train', target_train','sugeno',g_NF_clusterNum,opt);
epoch_n = 100;
dispOpt = zeros(1,4);
ANFIS_rmodel = anfis(Input_train0,in_fis,epoch_n,dispOpt);
end