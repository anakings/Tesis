function [] = distance_label3(name, N, antenna_type, PtdBm, frequency, r, p_matrix, azimuthFinalAngle, elevationFinalAngle)
   lambda = 3e8/frequency; % 3e8 is the light speed
%     r = lambda/2; % Radius of the circular antenna_type array
%     minimumPowerDB = -100; % Minimum transmission power in dBm
%     maximumPowerDB = -50; % Maximum transmission power in dBm
%     minimumPower = 10^((minimumPowerDB-30)/10); % Minimum transmission power in Watts
%     maximumPower = 10^((maximumPowerDB-30)/10); % Maximum transmission power in Watts
%     Pt = (maximumPower + minimumPower)/2; % Middle transmission power in Watts
    
    Pt = 10^((PtdBm-30)/10); % transmission power in Watts
    
    Gt = 1; % Transmitter gain (unity gain)
    L = 1; % System losses (lossless)

    %Gn = db2mag(patternAzimuth(antenna_type,frequency,(0:1:180))); % Directivity along the azimuth (dB)
    %Gn(:,end) = []; % Remove last column because the array has 361 elements (azimuth angles) instead of 360 (angles 0 and 360 are treated as different angles)
    %Gn(end,:) = []; % Remove last row because the array has 181 elements (elevation angles) instead of 180 (angles 0 and 180 are treated as different angles)
    
    % Calculate antenna_coordinates [Nx3]:
    antenna_coordinates = zeros(N+1, 3); 
    for n = 1:N
        antenna_coordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_coordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_coordinates(n,3) = 0; % z axis                               
    end
    antenna_coordinates(N+1,1) = 0;
    antenna_coordinates(N+1,2) = 0;
    antenna_coordinates(N+1,3) = r;
    
    
    for iter = 1:length(p_matrix)
        
        Pr = zeros(azimuthFinalAngle, elevationFinalAngle, N); % [360x180xN]
        % qua_vector = zeros(numOfExamples,N+4);
    
        p = p_matrix(iter); % Distance from source to center of system
        
        p_rand = p+rand();
    
        % Calculate the coordinates of the source:
        source_coordinates = zeros(azimuthFinalAngle, elevationFinalAngle, 3);
        for azimuthAngle = 1:azimuthFinalAngle                    %azimuth and elevation angle that form the source with the center of the system
            for elevationAngle = 1:elevationFinalAngle            %elevation angle that form the source with the center of the system
                source_coordinates(azimuthAngle,elevationAngle,1) = p_rand*cosd(azimuthAngle)*cosd(elevationAngle); % x axis
                source_coordinates(azimuthAngle,elevationAngle,2) = p_rand*sind(azimuthAngle)*cosd(elevationAngle); % y axis
                source_coordinates(azimuthAngle,elevationAngle,3) = p_rand*sind(elevationAngle); % z axis
            end
        end
    
        for elevationAngle = 1:elevationFinalAngle
            for azimuthAngle = 1:azimuthFinalAngle
                  
                Pr(azimuthAngle, elevationAngle, N+2) = p;
                
                x_t = source_coordinates(azimuthAngle,elevationAngle, 1);
                y_t = source_coordinates(azimuthAngle,elevationAngle, 2);
                z_t = source_coordinates(azimuthAngle,elevationAngle, 3);
                    
                if x_t < 0
                    if y_t < 0
                        qua = 3;
                    else
                        qua = 2;
                    end
                elseif y_t < 0 
                    qua = 4;
                else
                    qua = 1;
                end
                Pr(azimuthAngle, elevationAngle, N+3) = qua;
                    
%               qua_vector(i, 1)=qua;
%               qua_vector(i, 2)=x_t;
%               qua_vector(i, 3)=y_t;
%               qua_vector(i, 4)=z_t;
                    
                for n = 1:N+1
                    x_n = antenna_coordinates(n, 1);
                    y_n = antenna_coordinates(n, 2);
                    z_n = antenna_coordinates(n, 3);

                    d_n = sqrt(power(x_n-x_t, 2) + power(y_n-y_t, 2) + power(z_n-z_t, 2)); % Calculate the distance between the source and each of the antennas

                    % [-90:90]
                    elevation_n = asind((z_t - z_n)/d_n); % Calculate the elevation angle between the source and each of the antennas
                    
                    % Calculate the azimuth angle between the source and each of the antennas
                    y_len = abs(y_n - y_t);
                    x_len = abs(x_n - x_t);

                    if x_t == x_n && y_t == y_n
                        azimuth_n = 0;
                    elseif x_t > x_n && y_t >= y_n
                        azimuth_n = atand(y_len/x_len);
                    elseif x_t < x_n && y_t >= y_n
                        azimuth_n = 180-atand(y_len/x_len);
                    elseif x_t <= x_n && y_t < y_n
                        azimuth_n = 180+atand(y_len/x_len);
                    elseif x_t > x_n && y_t < y_n
                        azimuth_n = 360-atand(y_len/x_len);
                    elseif x_t == x_n && y_t < y_n
                        azimuth_n = 270;
                    elseif x_t == x_n && y_t > y_n
                        azimuth_n = 90;
                    end 
                    
                    azimuth_n = azimuth_n - 180; % [-180:180]
%                     
%                     if rem(n, 2) == 0   este if hace muy lento el codigo
%                         antenna_type = dipole;
%                     else
%                         antenna_type = dipoleVee;
                    Gn = db2mag(pattern(antenna_type, frequency, azimuth_n, elevation_n));
                    %Gn = 1.67*(cos(pi*cos(deg2rad(elevation_n-1))/2)/sin(deg2rad(elevation_n-1)).^2);
                    %Gn = 1.64*(cos(pi*cos(deg2rad(elevation_n-1))/2).^2/sin(deg2rad(elevation_n-1)).^2);
                    

                    Pr(azimuthAngle, elevationAngle, n) = Gn*(Pt*Gt*lambda^2/(d_n^2*16*pi^2*L)); % Calculate the received power in each of the antennas

                    %Pr(azimuthAngle, elevationAngle, n) = Gn(floor(azimuth_n)+1,floor(elevation_n)+1)*(Pt*Gt*lambda^2/(d_n^2*16*pi^2*L)); % Calculate the received power in each of the antennas
%                   qua_vector(i, 4+n)=Pr(i, n);
%                   qua_vector(i, 4+N+n)=Gn;
%                   qua_vector(i, 4+2N+n)=d_n;
                end
            end
        end
        
%         Prl = abs(Pr(:,:,(1:N+1))); % Friis space equation for received power along the operation distance [W]
%         row_sum = sum(Prl, 3);
%         Pr_normalized = Prl./row_sum;
%         Pr_normalized(:,:,N+2) = Pr(:,:,N+2);
%         Pr_normalized(:,:,N+3) = Pr(:,:,N+3);
%         
%         fileName = strcat(name, int2str(iter));
%         fileName = strcat(fileName, '_normalized.m');
% 
%         save(fileName, 'Pr_normalized');
        mass = iter;
        fileName = strcat(name, int2str(mass));
        fileName = strcat(fileName, '.m');

        save(fileName, 'Pr');
    end
end