clear;clc;

antenna_type = dipole;

iterations = 50;
 
frequency = 70e6;

PtdBm = 10; % Tx power given in dBm

folder = '/DOA_Datatest/';
mkdir([pwd folder]); % Create folder

azimuthFinalAngle = 360;
elevationFinalAngle = 180;

%%Create an array with random values of distance from transmitter to system
rng('default');

min_dist = 10;

%for max_dist = 1000:-200:200
for max_dist = 300
    p_matrix = zeros((max_dist-min_dist+1)*iterations, 1);
    min_index = 1;
    max_index = iterations;
    for distance = min_dist:max_dist
        p_matrix(min_index:max_index)= distance;
        min_index = max_index + 1;
        max_index = max_index + iterations;
    end
    
    for antennasNumber = 6:-2:4 % Generate data for a 4 to 16 antenna system
        mkdir([pwd folder int2str(max_dist)]);
        mkdir([pwd folder int2str(max_dist) '/' int2str(antennasNumber)]);
        name = strcat(pwd, folder, int2str(max_dist), '/', int2str(antennasNumber), '/iter_');
        distance_label(name, antennasNumber, antenna_type, PtdBm, frequency, p_matrix, azimuthFinalAngle, elevationFinalAngle);
    end
end




