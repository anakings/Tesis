clear;clc;

antenna_type = dipole;
%antenna_type = phased.IsotropicAntennaElement;

iterations = 10;
 
frequency = 70e6;
lambda = 3e8/frequency; % 3e8 is the light speed
r = lambda/2;

PtdBm = 10; % Tx power given in dBm

folder = '/dipole_antenna_plus/';
mkdir([pwd folder]); % Create folder

azimuthFinalAngle = 359;
elevationFinalAngle = 89;

%%Create an array with random values of distance from transmitter to system
rng('default');

min_dist = ceil(r/cosd(elevationFinalAngle));


for max_dist = min_dist + 90 % in meters
%for max_dist = 213 % in meters
    p_matrix = zeros((max_dist-min_dist+1)*iterations, 1);
    min_index = 1;
    max_index = iterations;
    for distance = min_dist:max_dist
        p_matrix(min_index:max_index)= distance;
        min_index = max_index + 1;
        max_index = max_index + iterations;
    end
    
    for antennasNumber = 4:4:8 % Generate data for a 4 to 16 antenna system
    %for antennasNumber = 4
        mkdir([pwd folder int2str(max_dist)]);
        mkdir([pwd folder int2str(max_dist) '/' int2str(antennasNumber)]);
        name = strcat(pwd, folder, int2str(max_dist), '/', int2str(antennasNumber), '/iter_');
        distance_label3(name, antennasNumber, antenna_type, PtdBm, frequency,r, p_matrix, azimuthFinalAngle, elevationFinalAngle);
    end
end




