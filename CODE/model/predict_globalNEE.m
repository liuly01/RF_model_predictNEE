close all;
clc
clear

bestRFModel = load('D:\han\writing\model\RF_predictNEP.mat');
BasePath='H';

%% sort out variables

% NDVI, T2M, T2M595, SSR, PRE, LCLAI, SOC, and CO2 as predictor variables
T2M595filepath = strcat(BasePath,':\predictNEE_VAR\2008\T2M0595\');
T2Mavefilepath = strcat(BasePath,':\predictNEE_VAR\2008\T2M250m\');
PREfilepath = strcat(BasePath,':\predictNEE_VAR\2008\PRE250m\');
SSRfilepath = strcat(BasePath,':\predictNEE_VAR\2008\SSR250m\');
NDVIfilepath = strcat(BasePath,':\predictNEE_VAR\2008\NDVI250m\');
LAIfilepath = strcat(BasePath,':\predictNEE_VAR\2008\LAI250m\');
SOCfilepath = 'D:\han\data\SOCC\250m\';

outputpath = strcat('G:\globalnee_product\han\2008\origin\');


T2M595filename = dir([T2M595filepath,'*.tif']);
T2Mavefilename = dir([T2Mavefilepath,'*.tif']);
PREfilename = dir([PREfilepath,'*.tif']);
SSRfilename = dir([SSRfilepath,'*.tif']);
NDVIfilename = dir([NDVIfilepath,'*.tif']);
LAIfilename = dir([LAIfilepath,'*.tif']);
SOCfilename = dir([SOCfilepath,'*.tif']);

%% predict global NEE
year = str2double(NDVIfilepath(19:22));
for i = 1:length(T2M595filepath)
    T2M595 = strcat(T2M595filepath, T2M595filename(i).name);
    T2Mave = strcat(T2Mavefilepath, T2Mavefilename(i).name);
    PRE = strcat(PREfilepath, PREfilename(i).name);
    SSR = strcat(SSRfilepath, SSRfilename(i).name);
    NDVI = strcat(NDVIfilepath, NDVIfilename(i).name);
    LAI = strcat(LAIfilepath, LAIfilename(i).name);
    SOC = strcat(SOCfilepath, SOCfilename(i).name);
    
    [T2M595data, T2M595R] = geotiffread(T2M595);
    [T2Mavedata, T2MaveR] = geotiffread(T2Mave);
    [PREdata, PRER] = geotiffread(PRE);
    [SSRdata, SSRR] = geotiffread(SSR);
    [NDVIdata, NDVIR] = geotiffread(NDVI);
    [LAIdata, LAIR] = geotiffread(LAI);
    [SOCdata, SOCR] = geotiffread(SOC);
    
    LAIdata(LAIdata==0) = nan;
    SOCdata = double(SOCdata);


    % process data to ensure dimensionality consistency
    col1 = size(T2M595data,2);
    col2 = size(NDVIdata,2);
    if col2>col1
        diff = col2-col1;
        T2M595data(:,col1:col1+diff) = T2M595data(:,col1-diff:col1);
        %PREdata(:,1:diff)= [];
        %SSRdata(:,1:diff)= [];
    elseif col2<col1
        diff = col1-col2;
        T2M595data(:,1:diff) = [];
    end
    raw1 = size(T2M595data,1);
    raw2 = size(NDVIdata,1);
    if raw2>raw1
        diff = raw2-raw1;
        T2M595data(raw1:raw1+diff,:) = T2M595data(raw1-diff:raw1,:);
        %PREdata(:,1:diff)= [];
        %SSRdata(:,1:diff)= [];
    elseif raw2<raw1
        diff = raw1-raw2;
        T2M595data(raw1-diff+1:raw1,:) = [];
    end
    
    % T2M
    col1 = size(T2Mavedata,2);
    if col2>col1
        diff = col2-col1;
        T2Mavedata(:,col1:col1+diff) = T2Mavedata(:,col1-diff:col1);
        %PREdata(:,1:diff)= [];
        %SSRdata(:,1:diff)= [];
    elseif col2<col1
        diff = col1-col2;
        T2Mavedata(:,1:diff) = [];
    end
    raw1 = size(T2Mavedata,1);
    if raw2>raw1
        diff = raw2-raw1;
        T2Mavedata(raw1:raw1+diff,:) = T2Mavedata(raw1-diff:raw1,:);
        %PREdata(:,1:diff)= [];
        %SSRdata(:,1:diff)= [];
    elseif raw2<raw1
        diff = raw1-raw2;
        T2Mavedata(raw1-diff+1:raw1,:) = [];
    end
    
    col1 = size(PREdata,2);
    if col2>col1
        diff = col2-col1;
        PREdata(:,col1:col1+diff) = PREdata(:,col1-diff:col1);
    elseif col2<col1
        diff = col1-col2;
        PREdata(:,1:diff) = [];
    end
    col1 = size(SSRdata,2);
    if col2>col1
        diff = col2-col1;
        SSRdata(:,col1:col1+diff) = SSRdata(:,col1-diff:col1);
    elseif col2<col1
        diff = col1-col2;
        SSRdata(:,1:diff) = [];
    end
    col1 = size(LAIdata,2);
    if col2>col1
        diff = col2-col1;
        LAIdata(:,col1:col1+diff) = LAIdata(:,col1-diff:col1);
    elseif col2<col1
        diff = col1-col2;
        LAIdata(:,1:diff) = [];
    end
    
    % cos(lat)
    [dataraw, datacol] = size(NDVIdata);
    
    % CO2
    % all pixels have the same value in the same year
    CO2list = [370.314 372.4772 374.6461 376.8219 379.001 381.187 383.3786...
        385.5772 387.779 389.9877 392.2022 394.4235 396.6481 398.8796...
        401.1167 403.3608 405.6082 407.8624 410.1223 412.388 414.7];
    CO2year = CO2list(year-2000);
    CO2data = ones(dataraw, datacol)*CO2year;
    
    
    % data preprocessing
    [Locationi,Locationj] = find(~isnan(T2M595data) & ~isnan(T2Mavedata) &~isnan(PREdata) & ~isnan(SSRdata) & ~isnan(NDVIdata) & ~isnan(LAIdata) & ~isnan(SOCdata));
    T2M595var = T2M595data(sub2ind(size(T2M595data),Locationi,Locationj));
    T2Mavevar = T2Mavedata(sub2ind(size(T2Mavedata),Locationi,Locationj));
    PREvar = PREdata(sub2ind(size(PREdata),Locationi,Locationj));
    SSRvar = SSRdata(sub2ind(size(SSRdata),Locationi,Locationj));
    clearvars T2M595data T2Mavedata PREdata SSRdata % release memory
    NDVIvar = NDVIdata(sub2ind(size(NDVIdata),Locationi,Locationj));
    LAIvar = LAIdata(sub2ind(size(LAIdata),Locationi,Locationj));
    CO2var = CO2data(sub2ind(size(CO2data),Locationi,Locationj));
    SOCvar = SOCdata(sub2ind(size(SOCdata),Locationi,Locationj));
    clearvars NDVIdata SOCdata CO2data LAIdata
    
    VAR = table(T2M595var,T2Mavevar,PREvar,SSRvar,NDVIvar,LAIvar,CO2var,SOCvar);
    VAR.Properties.VariableNames{1} = 'x1';
    VAR.Properties.VariableNames{2} = 'x2';
    VAR.Properties.VariableNames{3} = 'x3';
    VAR.Properties.VariableNames{4} = 'x4';
    VAR.Properties.VariableNames{5} = 'x5';
    VAR.Properties.VariableNames{6} = 'x6';
    VAR.Properties.VariableNames{7} = 'x7';
    VAR.Properties.VariableNames{8} = 'x8';
    n=size(T2M595var,1);
    clearvars T2M595var T2Mavevar PREvar SSRvar NDVIvar CO2var SOCvar LAIvar
%     clearvars T2M595var T2Mavevar PREvar SSRvar NDVIvar LAIvar SOCvar
    if n>100000000
        n1=round(n/3);
        n2=n1+round(n/3)+1;
        [Predict_label(1:n1,:),~] = predict(bestRFModel.bestRFModel, VAR(1:n1,:));
        [Predict_label(n1+1:n2,:),~] = predict(bestRFModel.bestRFModel, VAR(n1+1:n2,:));
        [Predict_label(n2+1:n,:),~] = predict(bestRFModel.bestRFModel, VAR(n2+1:n,:));
    else
        [Predict_label,~] = predict(bestRFModel.bestRFModel, VAR);
    end
    clearvars VAR
    
    result = [Locationi Locationj Predict_label];
    NEE=zeros(dataraw, datacol)*nan;
    NEE(sub2ind(size(NEE),result(:,1),result(:,2)))=result(:,3);
    % warning:must clear Predict_label
    clearvars Predict_label
    
    % Save predictions
    Refference=georasterref('RasterSize',size(NEE),'Latlim',NDVIR.LatitudeLimits,'Lonlim',NDVIR.LongitudeLimits);
	geotiffwrite(strcat(outputpath,num2str(year),'NEE',num2str(i),'.tif'),flipud(NEE),Refference);  
end