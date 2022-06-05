clear; clc;

step = 1; % Jumps between angles (Meanful angles)
azimuthStartAngle = -179;
azimuthFinalAngle = 180;
elevationStartAngle = 1;
elevationFinalAngle = 90;
totalAzimuthAngles = azimuthFinalAngle - azimuthStartAngle;
totalElevationAngles = elevationFinalAngle - elevationStartAngle;
totalAngles = totalAzimuthAngles*totalElevationAngles;
f1 = 70e6; % Signal frequency
fc = 70e6; % Receiver frequency
c = physconst('LightSpeed');
lambda = c/fc;
r = lambda/2; % Radio of the circular receiving system
fs = 1500; % Sample frequency
iterations = 1;
distance = 123; % Only to store the data in the same path where it is stored our proposal data
canShow = false; % To show (true) or not (false) the spectrum
nameFile = '/MUSIC_conformal_array'; % File name where the MUSIC results will be stored

% Array of different antennas configuration, based on the number of
% antennas used in the receiving system
sizeURAMatrix = [[2 2];[4 2];[4 3];[4 4]];
sizeURAMatrixSize = size(sizeURAMatrix);

for sizeURAIndex = 1:1:sizeURAMatrixSize(1)
    sizeURA = sizeURAMatrix(sizeURAIndex,:);
    N = sizeURA(1)*sizeURA(2); % Total number of antennas for the current array
    
    % Compute antennas coordinates [N+1 x 3],
    % N+1: Number of antennas, 1 => Antenna at the center of the system
    % 3 => x, y, z
    antenna_coordinates = zeros(N+1, 3); 
    for n = 1:N
        antenna_coordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_coordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_coordinates(n,3) = 0; % z axis                               
    end
    
    % Add antenna at the center of the receiving system
    antenna_coordinates(N+1,1) = 0;
    antenna_coordinates(N+1,2) = 0;
    antenna_coordinates(N+1,3) = r;

    act = antenna_coordinates.'; % act: Antenna coordinate transpose

    confarray = phased.ConformalArray('ElementPosition',[act(1,:); act(2,:); act(3,:)]);
    viewArray(confarray)
            
    % Array receiving frequency range
    minimumFreq = fc - 10e6;
    maximumFreq = fc + 10e6;
    array.Element.FrequencyRange = [minimumFreq maximumFreq];
    
    for iter = 1:1:iterations
        mkdir([pwd  '/results_SNR/' int2str(step) '/' int2str(distance) '/' 'multioutput_proposal' '/' int2str(N)  '/']); % Create folder
        name = strcat(pwd, '/results_SNR/', int2str(step), '/', int2str(distance), '/multioutput_proposal/', int2str(N), nameFile, int2str(iter), '.csv');

        for SNR = -10:10:40
            variance = 1 / 10.^(SNR/10);

            count_nan = [];
            count_wrong = 0; % Times the angle is found correctly
            count_well = 0; % Times the angle is found incorrectly
            azimuth_count_wrong = 0;
            elevation_count_wrong = 0;
            mseAzimuth = 0;
            mseElevation = 0;

            for azimuthAngle = azimuthStartAngle:step:azimuthFinalAngle
                for elevationAngle = elevationStartAngle:step:elevationFinalAngle
                    doa1 = [azimuthAngle; elevationAngle]; % Signal direction
                    
                    % Create signal
                    t = (0:1/fs:1).';
                    x1 = cos(2*pi*t*f1);
                    x = collectPlaneWave(confarray, x1, doa1, fc); % Received signal without noise
                    noise = variance*(randn(size(x)));
                    signal = x + noise;

                    % Create MUSIC estimator 2D
                    estimator = phased.MUSICEstimator2D('SensorArray', confarray, ...
                        'OperatingFrequency', fc, ...
                        'NumSignalsSource', 'Property', ...
                        'DOAOutputPort', true, 'NumSignals', 1, ...
                        'AzimuthScanAngles', azimuthStartAngle:step:azimuthFinalAngle, ...
                        'ElevationScanAngles', elevationStartAngle:step:elevationFinalAngle);
                     
                    [~,doas] = estimator(signal);
                    
                    azimuth_predict = doas(1);
                    elevation_predict = doas(2);
                    
                    if isnan(azimuth_predict) || isnan(elevation_predict)
                        count_nan_size = size(count_nan);
                        count_nan(count_nan_size(1) + 1,:) = doa1;
                    end
                    
                    % Compute wrong azimuth angle predictions and mse
                    [azimuth_count_wrong, mseAzimuth] = wrong(azimuth_predict, azimuthAngle, azimuth_count_wrong, mseAzimuth);
                    
                    % Compute wrong elevation angle predictions and mse
                    [elevation_count_wrong, mseElevation] = wrong(elevation_predict, elevationAngle, elevation_count_wrong, mseElevation);

                    % Set prediction as OK if both azimuth and elevation
                    % angle predictions are the same as the original angles
                    if azimuth_predict == azimuthAngle && elevation_predict == elevationAngle
                        count_well = count_well +1;
                    end

                    showSpectrum(canShow, estimator);
                end
            end
            
            % Compute accuracy
            accuracy = count_well/(totalAngles);
            accuracyAzimuth = (totalAngles-azimuth_count_wrong)/totalAngles;
            accuracyElevation = (totalAngles-elevation_count_wrong)/totalAngles;

            % Compute minimum square error
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
