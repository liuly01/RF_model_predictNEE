close all;
clc
clear

%% Data pre-collection
datafilename = 'D:\han\data\dataset2_aug.xlsx';
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
%     RFtree=[50,100,200,300];
%     RFLeaf=[3,5,10,20,50];
%     col='rgbcmyk';
%     figure('Name','RF Leaves and Trees');
%     for i=1:length(RFLeaf)
%         RFModel=TreeBagger(RFtree(RFOptimizationNum),data_var,data_label,'Method','R','OOBPrediction','On','MinLeafSize',RFLeaf(i));
%         plot(oobError(RFModel),col(i));
%         hold on
%     end
%     xlabel('Number of Grown Trees');
%     ylabel('Mean Squared Error') ;
%     LeafTreelgd=legend({'3' '5' '10' '20' '50'},'Location','NorthEast');
%     title(LeafTreelgd,'Number of Leaves');
%     hold off;
% 
%     disp(RFOptimizationNum);
% end


%% model train(5-fold Cross-validation)
realdatafilename = 'D:\han\writing\2024.9.24hqz_aug_corr\train data\dataset1_exaug.xlsx';
realdata = importdata(realdatafilename);
realneedata = realdata.data(:,9);
[M,N]=size(data_var); %size(data_nvar)
indices=crossvalind('Kfold',data_var(1:M,N),5); %data_nvar(1:M,N)
RFbias=[];

for i = 1
    test_reallabel = [];
    test_realvar = [];
    realRFPredict_label = [];
    realRFPredictConfidence = [];
    
    test = (indices == i); 
    train = ~test;
    train_var=data_var(train,:); %data_nvar(train,:)
    train_label=data_label(train,:); %data_nlabel(train,:)
    test_var=data_var(test,:); %data_nvar(test,:)
    test_label=data_label(test,:); %data_nlabel(test,:)
    
    %% RF model training and validation
    nTree=100;
    nLeaf=3;
    RFModel=TreeBagger(nTree,train_var,train_label,...
        'Method','regression','OOBPredictorImportance','on','MinLeafSize',nLeaf);
    
    % filter out the augmented data and exclusively use real data for model validation
    num = 1;
    for j = 1:length(test_label)
        if sum(ismember(realneedata,test_label(j,1))) ~= 0
            test_reallabel(num,1) = test_label(j,1);
            test_realvar(num,1:8) = test_var(j,1:8);
            num = num + 1;
        end
    end
    [realRFPredict_label,realRFPredictConfidence]=predict(RFModel,test_realvar);
    
    % real data Statistical indicators
    [realnum,~]=size(realRFPredict_label);
    realtest_mean = mean(test_reallabel);
    realpred_mean = mean(realRFPredict_label); 
    realRFerror = test_reallabel - realRFPredict_label;
    realRFbias(1,i) = sum(abs(realRFerror))/realnum;
    realRFR2(1,i) = (sum((test_reallabel-realtest_mean).*(realRFPredict_label-realpred_mean))/sqrt(sum((test_reallabel-realtest_mean).^2)*(sum((realRFPredict_label-realpred_mean).^2))))^2;
    realRFRMSE(1,i) = sqrt(sum(realRFerror.^2)/realnum);
    
    % anomalies optimise
    residual = test_reallabel-realRFPredict_label;
    residual(:,2) = (residual(:,1)-mean(residual(:,1)))/std(residual(:,1));
    residual_index = find(residual(:,2)>2|residual(:,2)<-2);

    opt_realRFPredict_label = realRFPredict_label;
    opt_realRFPredict_label(residual_index,:)=[];
    opt_test_reallabel = test_reallabel;
    opt_test_reallabel(residual_index,:)=[];
    opt_realRFPredictConfidence = realRFPredictConfidence;
    opt_realRFPredictConfidence(residual_index,:)=[];

    % anomalies optimise Statistical indicators
    [opt_realnum,~]=size(opt_realRFPredict_label);
    opt_realtest_mean = mean(opt_test_reallabel);
    opt_realpred_mean = mean(opt_realRFPredict_label); 
    opt_realRFerror = opt_test_reallabel - opt_realRFPredict_label;
    opt_realRFbias(1,i) = sum(abs(opt_realRFerror))/opt_realnum;
    opt_realRFR2(1,i) = (sum((opt_test_reallabel-opt_realtest_mean).*(opt_realRFPredict_label-opt_realpred_mean))/sqrt(sum((opt_test_reallabel-opt_realtest_mean).^2)*(sum((opt_realRFPredict_label-opt_realpred_mean).^2))))^2;
    opt_realRFRMSE(1,i) = sqrt(sum(opt_realRFerror.^2)/opt_realnum);
    
    % save model
    RFModelSavePath='D:\han\model\';
    save(sprintf('%sRF_Nee_dataaug.mat',RFModelSavePath),'nTree','nLeaf',...
        'RFModel','realRFPredictConfidence','realRFPredict_label','test_realvar','test_reallabel',...
        'realRFR2','realRFRMSE','realRFbias',...
        'opt_realRFR2','opt_realRFRMSE','opt_realRFbias',...
        'opt_realRFPredict_label','opt_test_reallabel','opt_realRFPredictConfidence');    
end  

%% statistical results
realbias = mean(realRFbias,2);
realR2 = mean(realRFR2,2);
realRMSE = mean(realRFRMSE,2);

optR2 = mean(opt_realRFR2,2);