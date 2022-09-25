clear; clc;

iterations = 1; % Number of matrixes to be generated for each distance
SNR_min = -10;
SNR_max = 40;
SNR_step = 10; % Jumps from SNR_min to SNR_max
SNR_list = SNR_min:SNR_step:SNR_max; % Usable SNRs
fc = 70e6; % Signal frequency
c = physconst('LightSpeed');
lambda = c/fc; % Wavelength
r = 10; % Radio of the circular receiving system
fs = 1500; % Sample frequency
withAntennaPlus = true; % Apply propagation loss (true) or not (false)

dipoleVeeLength = 143e6 / fc;
dipoleVeeArmLength = dipoleVeeLength / 2;
armElevationAngle = 45;
antenna_type = dipoleVee(...
    'ArmLength', [dipoleVeeArmLength, dipoleVeeArmLength], ...
    'ArmElevation', [armElevationAngle armElevationAngle]);
antenna_name = 'dipoleVee';

% Maximum linear dimension of antenna
D = 2 * dipoleVeeArmLength * sind(90 - armElevationAngle);

azimuthFinalAngle = 2;
elevationFinalAngle = 2;
angle_step = 1;

% Number of different distances
distanceQty = 0;

% Compute the minimum distance in meters between the source and the
% receiving system for far field
farFieldFirstCond = 2 * D^2 / lambda;
farFieldSecondCond = 10 * D;
farFieldThirdCond = 10 * lambda;
farFieldConditions = [...
    farFieldFirstCond ...
    farFieldSecondCond ...
    farFieldThirdCond];
min_dist = max(farFieldConditions);

% Create matrix of distances
for max_dist = min_dist + distanceQty
    d_matrix = zeros((max_dist-min_dist+1)*iterations, 1);
    min_index = 1;
    max_index = iterations;
    for distance = min_dist:max_dist
        d_matrix(min_index:max_index)= distance;
        min_index = max_index + 1;
        max_index = max_index + iterations;
    end
end

% Folder where the data matrix will be stored
if withAntennaPlus
    folder = strcat('/', antenna_name, '/data analytics/', '/correlation/', '/antenna plus/');
else
    folder = strcat('/', antenna_name, '/data analytics/', '/correlation/', '/without antenna plus/');
end

mkdir([pwd folder]); % Create folder

% Get antennas received power and save them on specified path
for antennasNumber = 16
    folder = strcat(folder, '/', int2str(antennasNumber), '/');
    mkdir([pwd folder]);
    plot_cov(folder, antennasNumber, antenna_type, fc, ...
        fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, angle_step, ...
        SNR_list, withAntennaPlus);
end