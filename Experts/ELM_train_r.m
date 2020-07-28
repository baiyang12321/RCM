function ELM_rmodel = ELM_train_r(Input_train_P, Input_train_T, ELMhidden_size)

[IW,B,LW,TF,TYPE] = elmtrain(Input_train_P, Input_train_T, ELMhidden_size, 'sig', 0);
ELM_rmodel={IW,B,LW,TF,TYPE};

end