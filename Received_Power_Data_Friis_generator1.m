clear;clc;

antenna_type = dipole;

iterations = 100;
 
frequency = 70e6;

PtdBm = 10; % Tx power given in dBm

folder = '/DOA_Data_test/antennas/';
mkdir([pwd folder]); % Create folder

%%Create an array with random values of distance from transmitter to system
rng('default');
p_matrix = zeros(iterations, 1);
min_dist = 10;
max_dist = 10;
% for i = 1:iterations
%     p_matrix(i) = (max_dist-min_dist)*rand() + min_dist;
% end
for i = 1:iterations
    p_matrix(i) = count;
    count = count+20
end

% max = 0
% for i = 1:iterations
%     p_matrix(i) = max_dist;
%     max = max + 1
%     if max == 100
%         max_dist = max_dist+10;
%         max = 0
%     end
% end

for antennasNumber = 2:5 % Generate data for a two to 16 antenna system
     mkdir([pwd folder int2str(antennasNumber) '/']);
     name = strcat(pwd, folder, int2str(antennasNumber), '/iter_');
     N = antennasNumber;
     Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix, PtdBm);
end

% folder = '/DOA_Data/distances/';
% mkdir([pwd folder]); % Create folder
% N=4
% for max_dist = 100:100:1000 % Generate data for a two to 16 antenna system
%     for i = 1:iterations
%         p_matrix(i) = (max_dist-min_dist)*rand() + min_dist;
%         p_matrix(i) = randi(max_dist)+min_dist;
%     end
%     mkdir([pwd folder int2str(max_dist) '/']);
%     name = strcat(pwd, folder, int2str(max_dist), '/iter_');
%     Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix, PtdBm);
% end


