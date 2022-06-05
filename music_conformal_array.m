clear;
clc;

step = 1;
azimuthStartAngle = -179;
azimuthFinalAngle = 180;
elevationStartAngle = 1;
elevationFinalAngle = 90;
totalAzimuthAngles = azimuthFinalAngle - azimuthStartAngle;
totalElevationAngles = elevationFinalAngle - elevationStartAngle;
totalAngles = totalAzimuthAngles*totalElevationAngles;
f1 = 70e6;
fc = 70e6;
c = physconst('LightSpeed');
lam = c/fc;
r = lam/2;
fs = 1500;
iterations = 1;
distance = 213; %esto es solo para guardar los datos de la misma forma que guardo los resultados de mi propuesta, o se usa para nada aqui
canShow = false;
nameFile = '/MUSIC_conformal_array';

sizeURAMatrix = [[2 2];[4 2];[4 3];[4 4]];
sizeURAMatrixSize = size(sizeURAMatrix);

for sizeURAIndex = 1:1:sizeURAMatrixSize(1)
    sizeURA = sizeURAMatrix(sizeURAIndex,:);
    N = sizeURA(1)*sizeURA(2);
    
    antenna_coordinates = zeros(N+1, 3); 
    for n = 1:N
        antenna_coordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_coordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_coordinates(n,3) = 0; % z axis                               
    end
    antenna_coordinates(N+1,1) = 0;
    antenna_coordinates(N+1,2) = 0;
    antenna_coordinates(N+1,3) = r;

    cc = antenna_coordinates(:,1);
    cc = antenna_coordinates.';

    %sCA = phased.ConformalArray('ElementPosition',[[2.1429 0 0 0];[0 2.1429 0 0];[-2.1429 0 0 2.1429]]);
    confarray = phased.ConformalArray('ElementPosition',[cc(1,:);cc(2,:);cc(3,:)]);
    viewArray(confarray)
            
    array.Element.FrequencyRange = [60e6 80e6];
    
    for iter = 1:1:iterations
        mkdir([pwd  '/results_SNR/' int2str(step) '/' int2str(distance) '/' 'multioutput_proposal' '/' int2str(N)  '/']); % Create folder
        name = strcat(pwd, '/results_SNR/', int2str(step), '/', int2str(distance), '/multioutput_proposal/', int2str(N), nameFile, int2str(iter), '.csv');

        for SNR = -10:10:40
            variance = 10.^(-8) / 10.^(SNR/10);

            count_nan = [];
            count_wrong = 0;
            count_well = 0;
            azimuth_count_wrong = 0;
            elevation_count_wrong = 0;
            mseAzimuth = 0;
            mseElevation = 0;

            for azimuthAngle = azimuthStartAngle:step:azimuthFinalAngle
                for elevationAngle = elevationStartAngle:step:elevationFinalAngle
                    doa1 = [azimuthAngle;elevationAngle];
                    

                    t = (0:1/fs:1).';
                    x1 = cos(2*pi*t*f1);
                    x = collectPlaneWave(confarray, x1, doa1, fc) * 1e-7;
                    noise = variance*(randn(size(x)));

                    estimator = phased.MUSICEstimator2D('SensorArray',confarray,...
                        'OperatingFrequency',fc,...
                        'NumSignalsSource','Property',...
                        'DOAOutputPort',true,'NumSignals',1,...
                        'AzimuthScanAngles',azimuthStartAngle:step:azimuthFinalAngle,...
                        'ElevationScanAngles',elevationStartAngle:step:elevationFinalAngle);
                    
                    signal = x + noise;
                    [~,doas] = estimator(x + noise);
                    
                    azimuth_predict = doas(1);
                    elevation_predict = doas(2);
                    
                    if isnan(azimuth_predict) || isnan(elevation_predict)
                        count_nan_size = size(count_nan);
                        count_nan(count_nan_size(1) + 1,:) = doa1;
                    end
                    
                    [azimuth_count_wrong, mseAzimuth] = wrong(azimuth_predict, azimuthAngle, azimuth_count_wrong, mseAzimuth);
                    
                    [elevation_count_wrong, mseElevation] = wrong(elevation_predict, elevationAngle, elevation_count_wrong, mseElevation);

                    if azimuth_predict == azimuthAngle && elevation_predict == elevationAngle
                        count_well = count_well +1;
                    end

                    showSpectrum(canShow, estimator);
                end
            end

            accuracy = count_well/(totalAngles);
            accuracyAzimuth = (totalAngles-azimuth_count_wrong)/totalAngles;
            accuracyElevation = (totalAngles-elevation_count_wrong)/totalAngles;

            mse = (mseAzimuth + mseElevation)/totalAngles;
            mseAzimuthFinal = mseAzimuth/totalAngles;
            mseElevationFinal = mseElevation/totalAngles;

            listSave = [N, distance, variance, accuracy, mse, accuracyAzimuth, accuracyElevation, mseAzimuthFinal, mseElevationFinal];

            writematrix(listSave, name,'WriteMode','append')
            
            count_nan
            count_nan_size = size(count_nan);
            count_nan_size(1)
        end
    end
end

function [countWrong, mseAngle] = wrong(predictAngle, realAngle, countWrong, mseAngle)
    if predictAngle ~= realAngle
        countWrong = countWrong + 1;
        if ~isnan(predictAngle)
            mseAngle = mseAngle + power(predictAngle-realAngle, 2);
        end
    end
end

function [] = showSpectrum(canShow, estimator)
    if canShow
        plotSpectrum(estimator);
    end
end
