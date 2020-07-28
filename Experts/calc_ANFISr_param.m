function [ best_cluster_num ] = calc_ANFISr_param(k, Input_train, Output_Data,DrawingSwitch)
  % 选择最佳的ANFIS参数（在这里是聚类数量）程序，结合生成的图表选择
  %
  % calc_ANFISr_param( k, Input_train0)
  %   k: k-折交叉验证
  %   Input_train0: 训练集数据, number_of_samples x (input_szie + 1)
  %
  % Example:
  %   calc_ANFISr_param(5,
  %               [0.480 0.3130	0.39700	0.1960	0.51000	0.28000	0.6360 1;
  %               0.447	0.2950	0.39900	0.8860	0.37200	0.18100	0.6430 2]）
  %
  % ================================================================
Input_train0=[Input_train;Output_Data]';
Indices=crossvalind('Kfold', size(Input_train0,1), k);
input_size = size(Input_train0,2) - 1;

x_axis = 2:2:30;
all_mean_MSE_val = zeros(length(x_axis),1);
all_mean_MSE_tra = zeros(length(x_axis),1);

all_mean_R2_val = zeros(length(x_axis),1);
all_mean_R2_tra = zeros(length(x_axis),1);

i = 0;
for cluster_n=x_axis
    i = i+1
    tic

    MSE_val = zeros(k, 1);
    MSE_tra = zeros(k, 1);

    R2s_val = zeros(k, 1);
    R2s_tra = zeros(k, 1);

%     val_input_cell = cell(k);
%     val_outpuy_cell = cell(k);
%     tra_input_cell = cell(k);
%     tra_output_cell = cell(k);
%     for n=1:k
%         validate_ind = Indices==n;
%         actTrain_ind = Indices~=n;
%         %%提取验证集
%         val_input_cell{n} = Input_train0(validate_ind, 1:input_size);
%         val_outpuy_cell{n} = Input_train0(validate_ind, input_size+1);
%         %%提取训练集
%         tra_input_cell{n} = Input_train0(actTrain_ind,1:input_size);
%         tra_output_cell{n} = Input_train0(actTrain_ind, input_size+1);
%     end
    parfor n = 1:k
        disp(['n=',num2str(n)]);
        validate_ind = Indices==n;
        actTrain_ind = Indices~=n;
        %%提取验证集
        validate_input = Input_train0(validate_ind, 1:input_size);
        validate_output = Input_train0(validate_ind, input_size+1);
        %%提取训练集
        actTrain_input = Input_train0(actTrain_ind,1:input_size);
        actTrain_output = Input_train0(actTrain_ind, input_size+1);
        trnData = [actTrain_input actTrain_output];
        %         mfType = 'gbellmf';
        opt = NaN(4,1);
        opt(4) = 0;
        in_fis = genfis3(actTrain_input,actTrain_output,'sugeno',cluster_n,opt);
        epoch_n = 100;
        dispOpt = zeros(1,4);
        out_fis = anfis(trnData,in_fis,epoch_n,dispOpt);
        try
            y_tra = evalfis(actTrain_input, out_fis);
            y_val = evalfis(validate_input, out_fis);

            perf_val = mse(y_val, validate_output);
            perf_tra = mse(y_tra, actTrain_output);

            r_val = corrcoef(y_val, validate_output);
            r2_val = prod(r_val(:));
            r_tra = corrcoef(y_tra, actTrain_output);
            r2_tra = prod(r_tra(:));
        catch ME
            disp([num2str(i),num2str(n),ME.identifier]);
            perf_val = 0;
            perf_tra = 0;
            r2_val = 1;
            r2_tra = 1;
        end
        MSE_val(n, 1) = perf_val;
        MSE_tra(n, 1) = perf_tra;

        R2s_val(n, 1) = r2_val;
        R2s_tra(n, 1) = r2_tra;
    end
    toc
    all_mean_MSE_val(i, 1) = mean(MSE_val);
    all_mean_MSE_tra(i, 1) = mean(MSE_tra);

    all_mean_R2_val(i, 1) = mean(R2s_val);
    all_mean_R2_tra(i, 1) = mean(R2s_tra);
end

if DrawingSwitch==1
figure(1)
[h1Ax,h1Line1,h1Line2] = plotyy(x_axis, all_mean_MSE_tra, x_axis, all_mean_R2_tra);
title('训练集')
xlabel('MF数量')
ylabel(h1Ax(1), 'MSE')
ylabel(h1Ax(2), 'R2')

figure(2)
[h2Ax,h2Line1,h2Line2] = plotyy(x_axis, all_mean_MSE_val, x_axis, all_mean_R2_val);
title('验证集')
xlabel('MF数量')
ylabel(h2Ax(1), 'MSE')
ylabel(h2Ax(2), 'R2')
end

[~, min_ind] = max(all_mean_R2_val);
best_cluster_num = x_axis(min_ind);
save calc_ANFIS_param.mat;
movefile('calc_ANFIS_param.mat', 'workfile');

end
