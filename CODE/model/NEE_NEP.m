clc;
clear;


datadir250 = 'G:\globalnee_product\han\2008\origin\';
NEEfilename250 = dir([datadir250,'*.tif']);

%% Generate an area of 250m grid
for i = 1:length(NEEfilename250)
    NEE250 = strcat(datadir250, NEEfilename250(i).name);
    [NEEdata, NEER] = geotiffread(NEE250);
    
    area_grid = zeros(size(NEEdata,1), size(NEEdata,2));
    % Get the starting latitude and longitude of the whole grid
    LAT = NEER.LatitudeLimits(1,2);
    LON = NEER.LongitudeLimits(1,1);
    % row cycle
    for j = 1:size(area_grid,1)
        % column cycle
        for k = 1:size(area_grid,2)
            lat1 = LAT-(j-1)*0.0022457882; % CellExtentInLatitude
            lon1 = LON-(k-1)*0.0022457882; % CellExtentInLongitude
            lat2 = LAT-j*0.0022457882;
            lon2 = LON-k*0.0022457882;
            area_grid(j,k) = pi/180*6371010.162^2*abs(sind(lat2)-sind(lat1))*abs(lon2-lon1);
        end
    end
    Refference=georasterref('RasterSize',size(area_grid),'Latlim',NEER.LatitudeLimits,'Lonlim',NEER.LongitudeLimits);
    geotiffwrite(strcat('D:\han\NEE_train\globalnee\area_reference250\72\area_reference',num2str(i),'.tif'),flipud(area_grid),Refference);
end

areadir250 = 'D:\han\NEE_train\globalnee\area_reference250\72\';
areafilename250 = dir([areadir250,'*.tif']);
for i = 1:length(NEE250)
    NEE250 = strcat(datadir250, NEEfilename250(i).name);
    outname = NEEfilename250(i).name;
    [NEEdata, NEER] = geotiffread(NEE250);

    area250 = strcat(areadir250, areafilename250(i).name);
    [areadata, areaR] = geotiffread(area250);
    
    NEEsum = NEEdata.*areadata;
    % gC - PgC
    NEEsum = NEEsum/10^15;
    NEEsum = NEEsum*-1;
    Refference=georasterref('RasterSize',size(NEEsum),'Latlim',NEER.LatitudeLimits,'Lonlim',NEER.LongitudeLimits);
    geotiffwrite(strcat('G:\globalnee_product\han\2008\gridsum\',outname(1:9),'_gridsum.tif'),flipud(NEEsum),Refference);
end