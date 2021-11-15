clear;clc;

antenna_type = dipole;

iterations = 500;
 
frequency = 70e6;

PtdBm = 10; % Tx power given in dBm

folder = '/DOA_Data_test1/';
mkdir([pwd folder]); % Create folder

%%Create an array with random values of distance from transmitter to system
rng('default');

min_dist = 10;

for max_dist = 400:-200:200
    p_matrix = zeros(iterations, 1);
    for i = 1:iterations
        p_matrix(i) = i*10;
    end
    
    for antennasNumber = 8:-2:4 % Generate data for a two to 16 antenna system
        mkdir([pwd folder int2str(max_dist)]);
        mkdir([pwd folder int2str(max_dist) '/' int2str(antennasNumber)]);
        name = strcat(pwd, folder, int2str(max_dist), '/', int2str(antennasNumber), '/iter_');
        %Received_Power_Data_Friis1(antennasNumber, frequency, iterations, name, antenna_type, p_matrix, PtdBm, max_dist,min_dist);
        Received_Power_Data_Friis1(antennasNumber, frequency, iterations, name, antenna_type, p_matrix, PtdBm);
    end
end




