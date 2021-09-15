function [] = Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix,PtdBm)
    lambda = 3e8/frequency; % 3e8 is the light speed
    r = lambda/2; % Radius of the circular antenna_type array
%     minimumPowerDB = -100; % Minimum transmission power in dBm
%     maximumPowerDB = -50; % Maximum transmission power in dBm
%     minimumPower = 10^((minimumPowerDB-30)/10); % Minimum transmission power in Watts
%     maximumPower = 10^((maximumPowerDB-30)/10); % Maximum transmission power in Watts
%     Pt = (maximumPower + minimumPower)/2; % Middle transmission power in Watts
    
    Pt = 10^((PtdBm-30)/10); % transmission power in Watts
    
    Gt = 1; % Transmitter gain (unity gain)
    L = 1; % System losses (lossless)

    Gn = db2mag(patternAzimuth(antenna_type,frequency,(0:1:180))); % Directivity along the azimuth
    %Gn = patternAzimuth(antenna_type,frequency,(0:1:180))
%     GndB = patternAzimuth(antenna_type,frequency,(0:1:180)); % Directivity along the azimuth
%     Gn = 10.^(GndB/10);
    Gn(:,end) = []; % Remove last column because the array has 361 elements (azimuth angles) instead of 360 (angles 0 and 360 are treated as different angles)
    Gn(end,:) = []; % Remove last row because the array has 181 elements (elevation angles) instead of 180 (angles 0 and 180 are treated as different angles)
    
    % Calculate antenna_cordinates [Nx3]:
    antenna_cordinates = zeros(N, 3); 
    for n = 1:N
        antenna_cordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_cordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_cordinates(n,3) = 0; % z axis
    end
    
    azimuthFinalAngle = 360;
    elevationFinalAngle = 180;
    for iter = 1:iterations
        iter
    
        p = p_matrix(iter); % Distance from source to center of system
    
        % Calculate the coordinates of the source:
        source_cordinates = zeros(azimuthFinalAngle, elevationFinalAngle, 3);
        for azimuthAngle = 1:azimuthFinalAngle
            for elevationAngle = 1:elevationFinalAngle
                source_cordinates(azimuthAngle,elevationAngle,1) = p*cosd(azimuthAngle-1)*cosd(elevationAngle-1); % x axis
                source_cordinates(azimuthAngle,elevationAngle,2) = p*sind(azimuthAngle-1)*cosd(elevationAngle-1);% y axis
                source_cordinates(azimuthAngle,elevationAngle,3) = p*sind(elevationAngle-1); % z axis
            end
        end
    
        Pr = zeros(azimuthFinalAngle, elevationFinalAngle, N); % [360x180xN]
        for n = 1:N
            x_n = antenna_cordinates(n, 1);
            y_n = antenna_cordinates(n, 2);
            z_n = antenna_cordinates(n, 3);
            for azimuthAngle = 1:azimuthFinalAngle-1
                for elevationAngle = 1:elevationFinalAngle-1
                    
                    x_t = source_cordinates(azimuthAngle+1,elevationAngle+1, 1);
                    y_t = source_cordinates(azimuthAngle+1,elevationAngle+1, 2);
                    z_t = source_cordinates(azimuthAngle+1,elevationAngle+1, 3);
                    
                    d_n = sqrt(power(x_n-x_t, 2) + power(y_n-y_t, 2) + power(z_n-z_t, 2)); % Calculate the distance between the source and each of the antennas
                    
                    elevation_n = round(asind(z_t/d_n)); % Calculate the elevation angle between the source and each of the antennas
                    
                    % Calculate the azimuth angle between the source and each of the antennas
                    y_len = abs(y_n - y_t);
                    x_len = abs(x_n - x_t);
                   
                    if x_t>=0 && y_t>=0
                       azimuth_n = round(rad2deg(atan(y_len/x_len)));
                    elseif x_t<0 && y_t>0
                       azimuth_n = round(180-rad2deg(atan(y_len/x_len)));
                    elseif x_t<0 && y_t<0
                       azimuth_n = round(180+rad2deg(atan(y_len/x_len)));
                    else
                       azimuth_n = round(360-rad2deg(atan(y_len/x_len)));
                    end
                    
                    if azimuth_n == 360
                        azimuth_n = 0;
                    end
  
                    Pr(azimuthAngle, elevationAngle, n) = Gn(azimuth_n+1, elevation_n+1)*(Pt*Gt*lambda^2/(d_n^2*16*pi^2*L)); % Calculate the received power in each of the antennas
                end
            end
        end
        
        Prl = abs(Pr); % Friis space equation for received power along the operation distance [W]
        row_sum = sum(Prl, 3);
        Pr_normalized = Pr./row_sum;
        
        fileName = strcat(name, int2str(iter));
        fileName = strcat(fileName, '_normalized.m');

        save(fileName, 'Pr_normalized');
    end
end