function [best_hidden_size] = calc_ELMr_param( k, Input_train, Output_Data,DrawingSwitch)
  % 选择最佳的BPNN参数（在这里是隐含层神经元数量）程序，结合生成的图表选择
  %
  % calc_BPNNr_param( k, train_set)
  %   k: k-折交叉验证
  %   train_set: 训练集数据, number_of_samples x (input_szie + 1)
  %
  % Example:
  %   calc_BPNNr_param(5,
  %               [0.480 0.3130	0.39700	0.1960	0.51000	0.28000	0.6360 1;
  %               0.447	0.2950	0.39900	0.8860	0.37200	0.18100	0.6430 2]）
  %
  % ================================================================
Input_train0=[Input_train;Output_Data]';
Indices=crossvalind('Kfold', size(Input_train0,1), k);
input_size = size(Input_train0,2) - 1;


hidden_size0=size(Input_train0,2)-3:30*size(Input_train0,2);%%设定隐含层步长
all_mean_MSE_val = zeros(size(hidden_size0,2),1);%%分配内存
all_mean_MSE_tra = zeros(size(hidden_size0,2),1);%%分配内存
all_mean_R2_val = zeros(size(hidden_size0,2),1);%%分配内存
all_mean_R2_tra = zeros(size(hidden_size0,2),1);%%分配内存
x_axis = size(Input_train0,2)-4:3*size(Input_train0,2);

%%生成优选数据集
HiddenSizeData=HiddenData(hidden_size0,1);

for hidden_size = 1:size(HiddenSizeData,1)
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
        
        
        [IW,B,LW,TF,TYPE] = elmtrain(actTrain_input, actTrain_output, HiddenSizeData(hidden_size,:),'sig',0);
        
        y_val = elmpredict(validate_input,IW,B,LW,TF,TYPE);
        y_tra = elmpredict(actTrain_input,IW,B,LW,TF,TYPE);
        
        [R2_val,MSE_val(i),RMSE_val,MAPE_val,MAD_val(i)]=predictorsfun(validate_output,y_val);
        [R2_tra,MSE_tra(i),RMSE_tra,MAPE_tra,MAD_tra(i)]=predictorsfun(actTrain_output,y_tra);

    end

    all_mean_MSE_val(hidden_size, 1) = mean(MSE_val);
    all_mean_MSE_tra(hidden_size, 1) = mean(MSE_tra);

    all_mean_MAD_val(hidden_size, 1) = mean(MAD_val);
    all_mean_MAD_tra(hidden_size, 1) = mean(MAD_tra);
    
    disp(['Hidden_size=[',num2str(HiddenSizeData(hidden_size,:)),']','    ','训练误差：',num2str(all_mean_MAD_tra(hidden_size,1)),'   ','验证误差：',num2str(all_mean_MAD_val(hidden_size, 1))]);
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
best_hidden_size = HiddenSizeData(min_ind,:);
save('calc_ELMr_param.mat');
movefile('calc_ELMr_param.mat', 'workfile');
end