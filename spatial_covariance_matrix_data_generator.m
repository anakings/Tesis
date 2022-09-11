clear; clc;

iterations = 1; % Number of matrixes to be generated for each distance
SNR_min = 10;
SNR_max = 20;
SNR_step = 10; % Jumps from SNR_min to SNR_max
SNR_list = SNR_min:SNR_step:SNR_max; % Usable SNRs
fc = 500e6; % Signal frequency
c = physconst('LightSpeed');
lambda = c/fc; % Wavelength
r = 0.75; % Radio of the circular receiving system
fs = 1024; % Sample frequency
withLoss = true; % Apply propagation loss (true) or not (false)

antenna_type = phased.IsotropicAntennaElement(...
    'FrequencyRange',[400e6 600e6],'BackBaffled',false);
antenna_name = 'isotropic';

azimuthFinalAngle = 360;
elevationFinalAngle = 90;
angle_step = 0.3;

% Number of different distances
distanceQty = 0;

% Compute the minimum distance in meters between the source and the
% receiving system for far field
min_dist = 10; %aqui la distancia no importa para nada

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
if withLoss
    folder = strcat('/', antenna_name, '/data_with_loss/');
else
    folder = strcat('/', antenna_name, '/data_no_loss/');
end

mkdir([pwd folder]); % Create folder

% Get antennas received power and save them on specified path
for antennasNumber = 9
    mkdir([pwd folder '/' int2str(antennasNumber)]);
    name = strcat(pwd, folder, '/', int2str(antennasNumber), '/iter_');
    spatial_covariance_matrix_data(name, antennasNumber, antenna_type, fc, lambda, ...
        fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, angle_step, ...
        SNR_list, withLoss);
end
