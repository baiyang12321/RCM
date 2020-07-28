function [RBFSpread, RBFMaxHiddenNerualNum] = calc_RBFr_param( k, Input_train, Output_Data,DrawingSwitch)
% 选择最佳的RBF参数（在这里是隐含层神经元数量）程序，结合生成的图表选择
  %
  % calc_RBFr_param( k, train_set)
  %   k: k-折交叉验证
  %   train_set: 训练集数据, number_of_samples x (input_szie + 1)
  %
  % Example:
  %   calc_RBFr_param(5,
  %               [0.480 0.3130	0.39700	0.1960	0.51000	0.28000	0.6360 1;
  %               0.447	0.2950	0.39900	0.8860	0.37200	0.18100	0.6430 2]）
  %
  % ================================================================
Input_train0=[Input_train;Output_Data]';
Indices=crossvalind('Kfold', size(Input_train0,1), k);
input_size = size(Input_train0,2) - 1;


Spread0 = 0.5:0.1:1.2;%%设定隐含层步长
all_mean_MSE_val = zeros(size(Spread0,2),1);%%分配内存
all_mean_MSE_tra = zeros(size(Spread0,2),1);%%分配内存
all_mean_R2_val = zeros(size(Spread0,2),1);%%分配内存
all_mean_R2_tra = zeros(size(Spread0,2),1);%%分配内存
x_axis = Spread0;

%%生成优选数据集
HiddenSizeData = Spread0';

for Spread_num = 1:size(Spread0,2)
    MSE_val = zeros(k, 1);
    MSE_tra = zeros(k, 1);

    MAD_val = zeros(k, 1);
    MAD_tra = zeros(k, 1);
    for i = 1:k
        validate_ind = Indices==i;
        actTrain_ind = Indices~=i;
        %%提取验证集
        validate_input = Input_train0(validate_ind, 1:input_size)';
        validate_output = Input_train0(validate_ind, input_size+1)';
        %%提取训练集
        actTrain_input = Input_train0(actTrain_ind,1:input_size)';
        actTrain_output = Input_train0(actTrain_ind, input_size+1)';

        net = newrb(actTrain_input, actTrain_output,0.001, HiddenSizeData(Spread_num,:), 35, 5);
        close all;
        y_val = sim(net, validate_input);
        y_tra = sim(net, actTrain_input);
        
        [R2_val,MSE_val(i),RMSE_val,MAPE_val,MAD_val(i)]=predictorsfun(validate_output,y_val);
        [R2_tra,MSE_tra(i),RMSE_tra,MAPE_tra,MAD_tra(i)]=predictorsfun(actTrain_output,y_tra);

    end

    all_mean_MSE_val(Spread_num, 1) = mean(MSE_val);
    all_mean_MSE_tra(Spread_num, 1) = mean(MSE_tra);

    all_mean_MAD_val(Spread_num, 1) = mean(MAD_val);
    all_mean_MAD_tra(Spread_num, 1) = mean(MAD_tra);
    
    disp(['Hidden_size=[',num2str(HiddenSizeData(Spread_num,:)),']','    ','训练误差：',num2str(all_mean_MAD_tra(Spread_num,1)),'   ','验证误差：',num2str(all_mean_MAD_val(Spread_num, 1))]);
end

if DrawingSwitch==1
figure(1)
[h1Ax,h1Line1,h1Line2] = plotyy(x_axis, all_mean_MSE_tra, x_axis, all_mean_R2_tra);
title('训练集')
xlabel('隐含层神经元数量')
ylabel(h1Ax(1), 'MSE')
ylabel(h1Ax(2), 'R2')

figure(2)
[h2Ax,h2Line1,h2Line2] = plotyy(x_axis, all_mean_MSE_val, x_axis, all_mean_R2_val);
title('验证集')
xlabel('隐含层神经元数量')
ylabel(h2Ax(1), 'MSE')
ylabel(h2Ax(2), 'R2')

end
[~, min_ind] = min(all_mean_MAD_val);
best_spread_size = HiddenSizeData(min_ind,:);
RBFSpread=best_spread_size;

%% 寻找最佳隐含层神经元数量
MaxHiddenNerualNum0 = 20:10:200;%%设定隐含层步长
all_mean_MSE_val = zeros(size(MaxHiddenNerualNum0,2),1);%%分配内存
all_mean_MSE_tra = zeros(size(MaxHiddenNerualNum0,2),1);%%分配内存
all_mean_R2_val = zeros(size(MaxHiddenNerualNum0,2),1);%%分配内存
all_mean_R2_tra = zeros(size(MaxHiddenNerualNum0,2),1);%%分配内存
x_axis = MaxHiddenNerualNum0;

%%生成优选数据集
MaxHiddenNerualData = MaxHiddenNerualNum0';

for MaxHiddenNerualNum = 1:size(MaxHiddenNerualNum0,2)
    MSE_val = zeros(k, 1);
    MSE_tra = zeros(k, 1);

    MAD_val = zeros(k, 1);
    MAD_tra = zeros(k, 1);
    for i = 1:k
        validate_ind = Indices==i;
        actTrain_ind = Indices~=i;
        %%提取验证集
        validate_input = Input_train0(validate_ind, 1:input_size)';
        validate_output = Input_train0(validate_ind, input_size+1)';
        %%提取训练集
        actTrain_input = Input_train0(actTrain_ind,1:input_size)';
        actTrain_output = Input_train0(actTrain_ind, input_size+1)';

        net = newrb(actTrain_input, actTrain_output,0.001, best_spread_size, MaxHiddenNerualData(MaxHiddenNerualNum,:), 5);
        close all;
        y_val = sim(net, validate_input);
        y_tra = sim(net, actTrain_input);
        
        [R2_val,MSE_val(i),RMSE_val,MAPE_val,MAD_val(i)]=predictorsfun(validate_output,y_val);
        [R2_tra,MSE_tra(i),RMSE_tra,MAPE_tra,MAD_tra(i)]=predictorsfun(actTrain_output,y_tra);

    end

    all_mean_MSE_val(MaxHiddenNerualNum, 1) = mean(MSE_val);
    all_mean_MSE_tra(MaxHiddenNerualNum, 1) = mean(MSE_tra);

    all_mean_MAD_val(MaxHiddenNerualNum, 1) = mean(MAD_val);
    all_mean_MAD_tra(MaxHiddenNerualNum, 1) = mean(MAD_tra);
    
    disp(['MaxHiddenNerualNum=[',num2str(MaxHiddenNerualData(MaxHiddenNerualNum,:)),']','    ','训练误差：',num2str(all_mean_MAD_tra(MaxHiddenNerualNum,1)),'   ','验证误差：',num2str(all_mean_MAD_val(MaxHiddenNerualNum, 1))]);
end

if DrawingSwitch==1
figure(1)
[h1Ax,h1Line1,h1Line2] = plotyy(x_axis, all_mean_MSE_tra, x_axis, all_mean_R2_tra);
title('训练集')
xlabel('隐含层神经元数量')
ylabel(h1Ax(1), 'MSE')
ylabel(h1Ax(2), 'R2')

figure(2)
[h2Ax,h2Line1,h2Line2] = plotyy(x_axis, all_mean_MSE_val, x_axis, all_mean_R2_val);
title('验证集')
xlabel('隐含层神经元数量')
ylabel(h2Ax(1), 'MSE')
ylabel(h2Ax(2), 'R2')

end
[~, min_ind] = min(all_mean_MAD_val);
RBFMaxHiddenNerualNum = MaxHiddenNerualData(min_ind,:);

save('calc_RBFr_param.mat');
movefile('calc_RBFr_param.mat', 'workfile');

end