clear;clc;

antenna_type = dipole;

iterations = 500;
 
frequency = 70e6;

PtdBm = 10; % Tx power given in dBm

folder = '/DOA_Data/antennas/';
mkdir([pwd folder]); % Create folder

% Create an array with random values of distance from transmitter to system
rng('default');
p_matrix = zeros(iterations, 1);
% min_dist = 10;
% max_dist = 100;
% for i = 1:iterations
%     p_matrix(i) = (max_dist-min_dist)*rand() + min_dist;
% end
for i = 1:iterations
    p_matrix(i) = randi(90)+10;
end

for antennasNumber = 4:16 % Generate data for a two to 16 antenna system
     mkdir([pwd folder int2str(antennasNumber) '/']);
     name = strcat(pwd, folder, int2str(antennasNumber), '/iter_');
     N = antennasNumber;
     Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix, PtdBm);
end

%N = 4;
%folder = '/DOA_Data/samples/';
%mkdir([pwd folder]);
%for samples = 100:100:500 %fix this
%    mkdir([pwd folder int2str(samples) '/']);
%    name = strcat(pwd, folder, int2str(samples), '/iter_');
%    iterations = samples;
%   Received_Power_Data_Friss(N, frequency, iterations, name, antenna_type);
