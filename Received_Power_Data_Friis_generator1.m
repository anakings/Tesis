clear;clc;

antenna_type = dipole;

iterations = 500;
 
frequency = 70e6;

folder = '/DOA_Data/antennas/';
mkdir([pwd folder]); % Create folder

% Create an array with random values of distance from transmitter to system
rng('default');
p_matrix = zeros(iterations, 1);
for i = 1:iterations
    p_matrix(i) = randi(100);
end

for antennasNumber = 2:16 % Generate data for a two to 16 antenna system
     mkdir([pwd folder int2str(antennasNumber) '/']);
     name = strcat(pwd, folder, int2str(antennasNumber), '/iter_');
     N = antennasNumber;
     Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix);
end

%N = 4;
%folder = '/DOA_Data/samples/';
%mkdir([pwd folder]);
%for samples = 100:100:500 %fix this
%    mkdir([pwd folder int2str(samples) '/']);
%    name = strcat(pwd, folder, int2str(samples), '/iter_');
%    iterations = samples;
%   Received_Power_Data_Friss(N, frequency, iterations, name, antenna_type);
