close all;
clc
clear

%% Data pre-collection
datafilename = 'D:\han\writing\train data\globalsite_traindata.xlsx';
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

%% Number of Leaves and Trees Optimization

% for RFOptimizationNum=1:4
%     
% RFLeaf=[3,5,10,20,50];
% col='rgbcmyk';
% figure('Name','RF Leaves and Trees');
% for i=1:length(RFLeaf)
%     RFModel=TreeBagger(2000,data_nvar,data_nlabel,'Method','R','OOBPrediction','On','MinLeafSize',RFLeaf(i));
%     plot(oobError(RFModel),col(i));
%     hold on
% end
% xlabel('Number of Grown Trees');
% ylabel('Mean Squared Error') ;
% LeafTreelgd=legend({'3' '5' '10' '20' '50'},'Location','NorthEast');
% title(LeafTreelgd,'Number of Leaves');
% hold off;
% 
% disp(RFOptimizationNum);
% end


%% model train(5-fold Cross-validation)
[M,N]=size(data_var); %size(data_nvar)
indices=crossvalind('Kfold',data_var(1:M,N),5); %data_nvar(1:M,N)
RFbias=[];

for i = 1:5
    test = (indices == i); 
    train = ~test;
    train_var=data_var(train,:); %data_nvar(train,:)
    train_label=data_label(train,:); %data_nlabel(train,:)
    test_var=data_var(test,:); %data_nvar(test,:)
    test_label=data_label(test,:); %data_nlabel(test,:)
    
    nTree=200;
    nLeaf=3;
    RFModel=TreeBagger(nTree,train_var,train_label,...
        'Method','regression','OOBPredictorImportance','on','cat',8:8,'MinLeafSize',nLeaf);
    [RFPredict_label,RFPredictConfidence]=predict(RFModel,test_var);

    
    % Statistical indicators
    [num,~]=size(RFPredict_label); %RFpred_label
    test_mean = mean(test_label);
    pred_mean = mean(RFPredict_label); %RFpred_label
    RFerror = test_label - RFPredict_label; %RFpred_label
    RFbias(1,i) = sum(abs(RFerror))/num;
     %RFpred_label
    RFR2(1,i) = (sum((test_label-test_mean).*(RFPredict_label-pred_mean))/sqrt(sum((test_label-test_mean).^2)*(sum((RFPredict_label-pred_mean).^2))))^2;
    RFRMSE(1,i) = sqrt(sum(RFerror.^2)/num);
    
    % Select the model with the largest R2
    if i==1
        bestRFbias = RFbias(1,1);
        bestRFRMSE = RFRMSE(1,1);
        bestRFR2 = RFR2(1,1);
        bestRFPredictConfidence = RFPredictConfidence;
        bestRFModel = RFModel;
        bestPredict_label = RFPredict_label;
        besttest_var = test_var;
        besttest_label = test_label;
        besttrain_var = train_var;
        besttrain_label = train_label;
    elseif RFR2(1,i)>bestRFR2
        bestRFbias = RFbias(1,i);
        bestRFRMSE = RFRMSE(1,i);
        bestRFR2 = RFR2(1,i);
        bestRFModel = RFModel;
        bestRFPredictConfidence = RFPredictConfidence;
        bestPredict_label = RFPredict_label;
        besttest_var = test_var;
        besttest_label = test_label;
        besttrain_var = train_var;
        besttrain_label = train_label;
    end
end

% statistical results
bias = mean(RFbias,2);
R2 = mean(RFR2,2);
RMSE = mean(RFRMSE,2);

%% save model
RFModelSavePath='D:\han\writing\model\';
save(sprintf('%sRF_predictNEP.mat',RFModelSavePath),'nTree','nLeaf',...
    'bestRFModel','bestRFPredictConfidence','bestPredict_label','bestRFR2','bestRFRMSE','bestRFbias',...
    'besttest_var','besttest_label','besttrain_var','besttrain_label');