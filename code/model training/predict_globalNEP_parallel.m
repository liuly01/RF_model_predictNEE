close all;
clc
clear

VARfilepath = 'F:\hqz2024\predictNEE_VAR\';
VARfileyear = dir(VARfilepath);

for VARyear = 3:24
    %% sort out variables
    BasePath = strcat(VARfilepath, VARfileyear(VARyear).name);
    T2M595filepath = strcat(BasePath, '\T2M0595\');
    T2Mavefilepath = strcat(BasePath, '\T2M250m\');
    PREfilepath = strcat(BasePath, '\PRE250m\');
    SSRfilepath = strcat(BasePath, '\SSR250m\');
    NDVIfilepath = strcat(BasePath, '\NDVI250m\');
    LAIfilepath = strcat(BasePath, '\LAI250m\');
    SOCfilepath = 'F:\hqz2024\predictNEE_VAR\SOCC\';

    outputpath = strcat('F:\hqz2024\output\', VARfileyear(VARyear).name, '\');

    T2M595filename = dir([T2M595filepath, '*.tif']);
    T2Mavefilename = dir([T2Mavefilepath, '*.tif']);
    PREfilename = dir([PREfilepath, '*.tif']);
    SSRfilename = dir([SSRfilepath, '*.tif']);
    NDVIfilename = dir([NDVIfilepath, '*.tif']);
    LAIfilename = dir([LAIfilepath, '*.tif']);
    SOCfilename = dir([SOCfilepath, '*.tif']);

    RFModel = load('F:\hqz2024\model\RF_Nee_dataaug_fold_exaug.mat');

    % Create parallel pool
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool('local', 3); % Specify the number of workers you want to use
    end

    %% predict global NEE
    year = str2double(NDVIfilepath(27:30));

    parfor i = 1:72
        % Broadcast constant variables into the loop
        T2M595 = strcat(T2M595filepath, T2M595filename(i).name);
        T2Mave = strcat(T2Mavefilepath, T2Mavefilename(i).name);
        PRE = strcat(PREfilepath, PREfilename(i).name);
        SSR = strcat(SSRfilepath, SSRfilename(i).name);
        NDVI = strcat(NDVIfilepath, NDVIfilename(i).name);
        LAI = strcat(LAIfilepath, LAIfilename(i).name);
        SOC = strcat(SOCfilepath, SOCfilename(i).name);

        [T2M595data, ~] = geotiffread(T2M595);
        [T2Mavedata, ~] = geotiffread(T2Mave);
        [PREdata, ~] = geotiffread(PRE);
        [SSRdata, ~] = geotiffread(SSR);
        [NDVIdata, ~] = geotiffread(NDVI);
        [LAIdata, ~] = geotiffread(LAI);
        [SOCdata, ~] = geotiffread(SOC);

        % CO2 data is broadcasted as it doesn't change per iteration
        CO2list = [370.314 372.4772 374.6461 376.8219 379.001 381.187 383.3786 385.5772 387.779 389.9877 392.2022 394.4235 396.6481 398.8796 401.1167 403.3608 405.6082 407.8624 410.1223 412.388 414.7];
        CO2year = CO2list(year - 2000);
        CO2data = ones(size(NDVIdata)) * CO2year;

        % Find non-NaN indices and prepare variables for prediction
        [Locationi, Locationj] = find(~isnan(T2M595data) & ~isnan(T2Mavedata) & ~isnan(PREdata) & ~isnan(SSRdata) & ~isnan(NDVIdata) & ~isnan(LAIdata) & ~isnan(SOCdata));

        VAR = table(T2M595data(sub2ind(size(T2M595data), Locationi, Locationj)), ...
                    T2Mavedata(sub2ind(size(T2Mavedata), Locationi, Locationj)), ...
                    PREdata(sub2ind(size(PREdata), Locationi, Locationj)), ...
                    SSRdata(sub2ind(size(SSRdata), Locationi, Locationj)), ...
                    NDVIdata(sub2ind(size(NDVIdata), Locationi, Locationj)), ...
                    LAIdata(sub2ind(size(LAIdata), Locationi, Locationj)), ...
                    CO2data(sub2ind(size(CO2data), Locationi, Locationj)), ...
                    SOCdata(sub2ind(size(SOCdata), Locationi, Locationj)));

        VAR.Properties.VariableNames = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'};

        % Predict using RFModel
        if size(VAR, 1) > 100000000
            % Split prediction if data is too large
            n = size(VAR, 1);
            n1 = round(n / 3);
            n2 = n1 + round(n / 3) + 1;
            Predict_label(1:n1, :) = predict(RFModel.RFModel, VAR(1:n1, :));
            Predict_label(n1+1:n2, :) = predict(RFModel.RFModel, VAR(n1+1:n2, :));
            Predict_label(n2+1:n, :) = predict(RFModel.RFModel, VAR(n2+1:n, :));
        else
            [Predict_label, Predict_confidence] = predict(RFModel.RFModel, VAR);
        end

        % Create NEE matrix
        NEE = nan(size(NDVIdata));
        result = [Locationi, Locationj, Predict_label];
        NEE(sub2ind(size(NEE), result(:, 1), result(:, 2))) = result(:, 3);
        
        NEE = single(NEE);

        % Save the NEE data
        reffilename = dir(['F:\hqz2024\reference\', '*.tif']);
        [~, R1] = geotiffread(strcat('F:\hqz2024\reference\', reffilename(i).name));

        Refference=georasterref('RasterSize',size(NEE),'Latlim',R1.LatitudeLimits,'Lonlim',R1.LongitudeLimits);
        geotiffwrite(strcat(outputpath,'origin/',num2str(year),'NEE',num2str(i),'.tif'),flipud(NEE),Refference);

    end
end