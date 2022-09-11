clear; clc;

step = 1;
azimuthStartAngle = -179;
azimuthFinalAngle = 180;
elevationStartAngle = 1;
elevationFinalAngle = 90;
f1 = 70e6;
fc = 70e6;
c = physconst('LightSpeed');
lam = c/fc;
r = lam/2;
fs = 1500;
iterations = 1;
dipoleVeeLength = 143e6 / fc;
dipoleVeeArmLength = dipoleVeeLength / 2;
armElevationAngle = 45;
antenna_type = dipoleVee(...
    'ArmLength', [dipoleVeeArmLength, dipoleVeeArmLength], ...
    'ArmElevation', [armElevationAngle armElevationAngle]);
antenna_name = 'dipoleVee';
canShowArray = false;
fileName = 'MUSIC_conformal_validation_with_same_angles_that_ML_model';

sizeURAMatrix = [[2 2];[3 2];[4 2];[5 2];[4 3];[7 2];[4 4]];

sizeURAMatrixSize = size(sizeURAMatrix);

% Create signal
t = (0:1/fs:1).';
x1 = cos(2*pi*t*f1); % Signal

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

    act = antenna_coordinates.';

    % Create receiving system
    confarray = phased.ConformalArray(...
        'Element', antenna_type, ...
        'ElementPosition', [act(1,:); act(2,:); act(3,:)]);
    %viewArray(confarray)
            
%     array.Element.FrequencyRange = [60e6 80e6];
    
    for iter = 1:1:iterations
        mkdir([pwd '/' antenna_name '/results/music/' int2str(step) ...
            '/Multioutput/' int2str(N)  '/']); % Create folder
        name = strcat(pwd,  '/', antenna_name, '/results/music/', ...
            int2str(step), '/Multioutput/', int2str(N), '/', fileName, ...
            int2str(iter), '.csv');

        for SNR = -10:10:40
            variance = 1 / 10.^(SNR/10);

            count_nan = [];
            count_wrong = 0;
            count_well = 0;
            azimuth_count_wrong = 0;
            elevation_count_wrong = 0;
            mseAzimuth = 0;
            mseElevation = 0;

            fileAzimuthValidation = ['./', antenna_name, '/', ...
                'validation_angles/1/Multioutput/', int2str(N), ...
                '/azimuthLabel', int2str(SNR), '.0.csv'];
            azimuthValidation = readmatrix(fileAzimuthValidation);
        
            fileElevationValidation = ['./', antenna_name, '/', ...
                'validation_angles/1/Multioutput/', int2str(N), ...
                '/elevationLabel', int2str(SNR), '.0.csv'];
            elevationValidation = readmatrix(fileElevationValidation);
            
            sizeAngles = size(elevationValidation);
            totalAngles = sizeAngles(2);
        
            for item = 1:1:totalAngles
                % It has to scale the azimuth angles from -180 to 180 
                % because those are the angles that collectPlaneWave works 
                % for
                azimuthAngle = azimuthValidation(item)-180;
                
                elevationAngle = elevationValidation(item);
                
                doa = [azimuthAngle; elevationAngle];

                x = collectPlaneWave(confarray, x1, doa, fc);
                noise = sqrt(variance) * randn(size(x));
                signal = x + noise;

                estimator = phased.MUSICEstimator2D(...
                    'SensorArray', confarray,...
                    'OperatingFrequency', fc, ...
                    'NumSignalsSource', 'Property', ...
                    'DOAOutputPort', true, 'NumSignals', 1, ...
                    'AzimuthScanAngles', ...
                    azimuthStartAngle:step:azimuthFinalAngle,...
                    'ElevationScanAngles', ...
                    elevationStartAngle:step:elevationFinalAngle);
                
                [~,doas] = estimator(x + noise);
                    
                azimuth_predict = doas(1);
                elevation_predict = doas(2);
                    
                if isnan(azimuth_predict) || isnan(elevation_predict)
                    count_nan_size = size(count_nan);
                    count_nan(count_nan_size(1) + 1,:) = doa;
                end
                    
                [azimuth_count_wrong, mseAzimuth] = wrong(...
                    azimuth_predict, azimuthAngle, ...
                    azimuth_count_wrong, mseAzimuth);
                    
                [elevation_count_wrong, mseElevation] = wrong(...
                    elevation_predict, elevationAngle, ...
                    elevation_count_wrong, mseElevation);

                if azimuth_predict == azimuthAngle && ...
                        elevation_predict == elevationAngle
                    count_well = count_well +1;
                end

                showSpectrum(canShowArray, estimator);
            end

            accuracy = count_well/(totalAngles);
            accuracyAzimuth = (totalAngles-azimuth_count_wrong)/...
                totalAngles;
            accuracyElevation = (totalAngles-elevation_count_wrong)/...
                totalAngles;
            
            count_nan
            count_nan_size = size(count_nan);
            count_nan_size = count_nan_size(1)

            mse = (mseAzimuth + mseElevation)/(totalAngles-count_nan_size);
            mseAzimuthFinal = mseAzimuth/(totalAngles-count_nan_size);
            mseElevationFinal = mseElevation/(totalAngles-count_nan_size);

            listSave = [N, SNR, accuracy, mse, ...
                accuracyAzimuth, accuracyElevation, mseAzimuthFinal, ...
                mseElevationFinal];

            writematrix(listSave, name,'WriteMode','append')
            
        end
    end
end

function [countWrong, mseAngle] = wrong(predictAngle, realAngle, ...
    countWrong, mseAngle)
    if predictAngle ~= realAngle
        countWrong = countWrong + 1;
        if ~isnan(predictAngle)
            mseAngle = mseAngle + power(predictAngle-realAngle, 2);
        end
    end
end

function [] = showSpectrum(canShowArray, estimator)
    if canShowArray
        plotSpectrum(estimator);
    end
end
