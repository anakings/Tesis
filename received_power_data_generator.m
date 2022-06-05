clear; clc;

antenna_type = dipoleVee;

iterations = 20; % Number of matrixes to be generated for each distance
SNR_min = -10;
SNR_max = 40;
SNR_step = 10; % Jumps from SNR_min to SNR_max
SNR_list = SNR_min:SNR_step:SNR_max; % Usable SNRs
 
frequency = 70e6; % Signal frequency
c = physconst('LightSpeed');
lambda = c/frequency; % Wavelength
r = lambda/2; % Radio of the circular receiving system
fs = 1500; % Sample frequency

folder = '/new_Data/'; % Folder where the data matrix will be stored
mkdir([pwd folder]); % Create folder

azimuthFinalAngle = 360;
elevationFinalAngle = 90;

% Number of different distances
distanceQty = 0;

% Minimum distance in meters between the source and the receiving system
min_dist = ceil(r/cosd(elevationFinalAngle-1));

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

% Get antennas received power and save them on specified path
for antennasNumber = 3:1:12
    mkdir([pwd folder '/' int2str(antennasNumber)]);
    name = strcat(pwd, folder, '/', int2str(antennasNumber), '/iter_');
    received_power_data(name, antennasNumber, antenna_type, frequency, fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, SNR_list);
end
