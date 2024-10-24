close all;
clc
clear

load('D:\han\model\RF_Nee_aug.mat')

%% filter out aug data
realdatafilename = 'D:\han\data\dataset1_exaug.xlsx';
realdata = importdata(realdatafilename);
realneedata = realdata.data(:,9);

train_reallabel = [];
train_realvar = [];
realRFPredict_label = [];
realRFPredictConfidence = [];

train_var=RFModel.X; %data_nvar(train,:)
train_label=RFModel.Y; %data_nlabel(train,:)

num = 1;
for j = 1:length(train_label)
    if sum(ismember(realneedata,train_label(j,1))) ~= 0
        train_reallabel(num,1) = train_label(j,1);
        train_realvar(num,1:8) = train_var(j,1:8);
        num = num + 1;
    end
end

%% filter opt data
num = 1;
for j = 1:length(opt_test_reallabel)
    if sum(ismember(test_reallabel,opt_test_reallabel(j,1))) ~= 0
        final_test_reallabel(num,1) = test_reallabel(find(ismember(test_reallabel,opt_test_reallabel(j,1))),1);
        final_test_realvar(num,1:8) = test_realvar(find(ismember(test_reallabel,opt_test_reallabel(j,1))),1:8);
        num = num + 1;
    end
end

%% RF model training and validation
nTree=100;
nLeaf=3;
RFModel=TreeBagger(nTree,train_realvar,train_reallabel,...
    'Method','regression','OOBPredictorImportance','on','MinLeafSize',nLeaf);

[realRFPredict_label,realRFPredictConfidence]=predict(RFModel,final_test_realvar);

[realnum,~]=size(realRFPredict_label);
realtest_mean = mean(final_test_reallabel);
realpred_mean = mean(realRFPredict_label); 
realRFerror = final_test_reallabel - realRFPredict_label;
realRFbias = sum(abs(realRFerror))/realnum;
realRFR2 = (sum((final_test_reallabel-realtest_mean).*(realRFPredict_label-realpred_mean))/sqrt(sum((final_test_reallabel-realtest_mean).^2)*(sum((realRFPredict_label-realpred_mean).^2))))^2;
realRFRMSE = sqrt(sum(realRFerror.^2)/realnum);

% save model
RFModelSavePath='D:\han\model\';
save(sprintf('%sRF_Nee_exaug.mat',RFModelSavePath),'nTree','nLeaf',...
    'RFModel','realRFPredictConfidence','realRFPredict_label','final_test_realvar','final_test_reallabel',...
    'realRFR2','realRFRMSE','realRFbias'); 