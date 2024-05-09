close all;
clc
clear

%% Data pre-collection
datafilename = 'D:\han\writing\2024.3.23hqz_scientific_data\train data\globalsite_traindata_excluding_aug.xlsx';
data = importdata(datafilename);
data_var = []; % variable
data_var(:,1) = data.data(:,14); % 595T2M
data_var(:,2) = data.data(:,15); % aveT2M
data_var(:,3) = data.data(:,16); % avePRE
data_var(:,4) = data.data(:,17); % aveSSR
data_var(:,5) = data.data(:,26); % NDVI
data_var(:,6) = data.data(:,24); % LAI
data_var(:,7) = data.data(:,25); % CO2
data_var(:,8) = data.data(:,28); % SOC


data_label = []; % nee
data_label(:,1) = data.data(:,9); % NEE


%% model train(5-fold Cross-validation)
[M,N]=size(data_var); %size(data_nvar)
indices=crossvalind('Kfold',data_var(1:M,N),10); %data_nvar(1:M,N)
RFbias=[];

for i = 1:10
    test_reallabel = [];
    test_realvar = [];
    RFPredict_label = [];
    RFPredictConfidence = [];
    
    test = (indices == i); 
    train = ~test;
    train_var=data_var(train,:); %data_nvar(train,:)
    train_label=data_label(train,:); %data_nlabel(train,:)
    test_var=data_var(test,:); %data_nvar(test,:)
    test_label=data_label(test,:); %data_nlabel(test,:)
    
    %% RF model training and validation
    nTree=200;
    nLeaf=3;
    RFModel=TreeBagger(nTree,train_var,train_label,...
        'Method','regression','OOBPredictorImportance','on','MinLeafSize',nLeaf);
    
    [RFPredict_label,RFPredictConfidence]=predict(RFModel,test_var);
    
    % real data Statistical indicators
    [num,~]=size(RFPredict_label);
    test_mean = mean(test_reallabel);
    pred_mean = mean(RFPredict_label); 
    RFerror = test_label - RFPredict_label;
    RFbias(1,i) = sum(abs(RFerror))/num;
    RFR2(1,i) = (sum((test_label-test_mean).*(RFPredict_label-pred_mean))/sqrt(sum((test_label-test_mean).^2)*(sum((RFPredict_label-pred_mean).^2))))^2;
    RFRMSE(1,i) = sqrt(sum(RFerror.^2)/num);
    
    % anomalies optimise
    residual = test_label-RFPredict_label;
    residual(:,2) = (residual(:,1)-mean(residual(:,1)))/std(residual(:,1));
    residual_index = find(residual(:,2)>2|residual(:,2)<-2);

    opt_RFPredict_label = RFPredict_label;
    opt_RFPredict_label(residual_index,:)=[];
    opt_test_label = test_label;
    opt_test_label(residual_index,:)=[];
    opt_RFPredictConfidence = RFPredictConfidence;
    opt_RFPredictConfidence(residual_index,:)=[];

    % anomalies optimise Statistical indicators
    [opt_num,~]=size(opt_RFPredict_label);
    opt_test_mean = mean(opt_test_label);
    opt_pred_mean = mean(opt_RFPredict_label); 
    opt_RFerror = opt_test_label - opt_RFPredict_label;
    opt_RFbias(1,i) = sum(abs(opt_RFerror))/opt_num;
    opt_RFR2(1,i) = (sum((opt_test_label-opt_test_mean).*(opt_RFPredict_label-opt_pred_mean))/sqrt(sum((opt_test_label-opt_test_mean).^2)*(sum((opt_RFPredict_label-opt_pred_mean).^2))))^2;
    opt_RFRMSE(1,i) = sqrt(sum(opt_RFerror.^2)/opt_num);
    
    % save model
    RFModelSavePath='D:\han\writing\2024.4.12hqz_scientific_data\model\experiment1\real data\';
    save(sprintf('%sRF_Nee_data_fold%d.mat',RFModelSavePath,i),'nTree','nLeaf',...
        'RFModel','RFPredictConfidence','RFPredict_label','test_var','test_label',...
        'RFR2','RFRMSE','RFbias',...
        'opt_RFR2','opt_RFRMSE','opt_RFbias',...
        'opt_RFPredict_label','opt_test_label','opt_RFPredictConfidence');    
end  

%% statistical results
realbias = mean(RFbias,2);
realR2 = mean(RFR2,2);
realRMSE = mean(RFRMSE,2);