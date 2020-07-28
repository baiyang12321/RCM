function RBF_model = RBF_train_r(input_train, target_train , Spread , NerualNum)

[RBF_model,tr]=newrb(input_train,target_train,0.001,Spread,NerualNum,5);
close all;
end