function [best_c, best_gamma ] = calc_SVMr_param(k, Input_train, Output_Data,DrawingSwitch)
  % 选择最佳的SVM参数（在这里是c,gamma）程序，结合生成的图表选择
  %
  % calc_ANFISr_param( k, Input_train0)
  %   k: k-折交叉验证
  %   Input_train0: 训练集数据, number_of_samples x (input_szie + 1)
  %
  % Example:
  %   calc_SVMr_param(5,
  %               [0.480 0.3130	0.39700	0.1960	0.51000	0.28000	0.6360 1;
  %               0.447	0.2950	0.39900	0.8860	0.37200	0.18100	0.6430 2]）
  %
  % ================================================================
Input_train0=[Input_train;Output_Data]';
Indices=crossvalind('Kfold', size(Input_train0,1), k);
input_size = size(Input_train0,2) - 1;

x_axis = -5:12;
y_axis = -5:5;
all_mean_MSE_val = zeros(length(x_axis),length(y_axis));
all_mean_MSE_tra = zeros(length(x_axis),length(y_axis));

all_mean_R2_val = zeros(length(x_axis),length(y_axis));
all_mean_R2_tra = zeros(length(x_axis),length(y_axis));

input_set = Input_train0(:, 1:input_size);
output_set = Input_train0(:, input_size+1);

i = 0;
for c_e = x_axis
    c = 2^c_e
    i = i+1;
    j = 0;
    for gamma_e = y_axis
        gamma = 2^gamma_e
        j=j+1;
        fprintf('%d,%d\n',i,j);
        MSE_val = zeros(k, 1);
        MSE_tra = zeros(k, 1);

        R2s_val = zeros(k, 1);
        R2s_tra = zeros(k, 1);

        parfor n = 1:k
            validate_ind = Indices==n;
            actTrain_ind = Indices~=n;
           %%提取验证集
            validate_input = input_set(validate_ind,:);
            validate_output = output_set(validate_ind, :);
            %%提取训练集
            actTrain_input = input_set(actTrain_ind,:);
            actTrain_output = output_set(actTrain_ind, :);
            %             actInput_train0 = Input_train0(actTrain_ind,:);

            SVM_model = fitrsvm(actTrain_input,actTrain_output, 'BoxConstraint',c,'KernelFunction','gaussian','KernelScale',gamma)
            y_tra = SVM_model.predict(actTrain_input);
            y_val = SVM_model.predict(validate_input);

            perf_val = mse(y_val, validate_output);
            perf_tra = mse(y_tra, actTrain_output);

            r_val = corrcoef(y_val, validate_output);
            r2_val = prod(r_val(:));
            r_tra = corrcoef(y_tra, actTrain_output);
            r2_tra = prod(r_tra(:));

            MSE_val(n, 1) = perf_val;
            MSE_tra(n, 1) = perf_tra;

            R2s_val(n, 1) = r2_val;
            R2s_tra(n, 1) = r2_tra;
        end

        all_mean_MSE_val(i, j) = mean(MSE_val);
        all_mean_MSE_tra(i, j) = mean(MSE_tra);

        all_mean_R2_val(i, j) = mean(R2s_val);
        all_mean_R2_tra(i, j) = mean(R2s_tra);
    end
end

if DrawingSwitch==1
figure(1)
contour(x_axis,y_axis,all_mean_MSE_tra')
title('训练集-MSE')
xlabel('C')
ylabel('gamma')

figure(2)
contour(x_axis,y_axis,all_mean_R2_tra')
title('选联机-R2')
xlabel('C')
ylabel('gamma')

figure(3)
contour(x_axis,y_axis,all_mean_MSE_val')
title('测试集-MSE')
xlabel('C')
ylabel('gamma')

figure(4)
contour(x_axis,y_axis,all_mean_R2_val')
title('测试集-R2')
xlabel('C')
ylabel('gamma')
end

M = max(all_mean_R2_val(:));
[iM,jM]= find(all_mean_R2_val==M);
best_c = x_axis(iM);
best_gamma = y_axis(jM);

save('calc_SVMr_param.mat');
movefile('calc_SVMr_param.mat', 'workfile');
end
