function [] = Received_Power_Data_Friis1(N, frequency, iterations, name, antenna_type, p_matrix)
    lambda = 3e8/frequency; % 3e8 is the light speed
    r = lambda/2; % Radius of the circular antenna_type array
    minimumPowerDB = -100; % Minimum transmission power in dBm
    maximumPowerDB = -50; % Maximum transmission power in dBm
    minimumPower = 10^((minimumPowerDB-30)/10); % Minimum transmission power in Watts
    maximumPower = 10^((maximumPowerDB-30)/10); % Maximum transmission power in Watts
    Pt = (maximumPower + minimumPower)/2; % Middle transmission power in Watts
    
    Gt = 1; % Transmitter gain (unity gain)
    L = 1; % System losses (lossless)

    %Gn = db2mag(patternAzimuth(antenna_type,frequency,(0:1:180))) % Directivity along the azimuth
    %patternAzimuth(antenna_type,frequency,(0:1:180))
    Gn = patternAzimuth(antenna_type,frequency,(0:1:180)); % Directivity along the azimuth
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
            for azimuthAngle = 1:azimuthFinalAngle
                for elevationAngle = 1:elevationFinalAngle
                    x_t = source_cordinates(azimuthAngle,elevationAngle, 1);
                    y_t = source_cordinates(azimuthAngle,elevationAngle, 2);
                    z_t = source_cordinates(azimuthAngle,elevationAngle, 3);
                    d = sqrt(power(x_n-x_t, 2) + power(y_n-y_t, 2) + power(z_n-z_t, 2)); % Calculate the distance between the source and each of the antennas
                    elevation = round(asind(z_t/d) + 1); % Calculate the elevation angle between the source and each of the antennas
                    Pr(azimuthAngle, elevationAngle, n) = Gn(azimuthAngle, elevation)*(Pt*Gt*lambda^2/(d^2*16*pi^2*L)); % Calculate the received power in each of the antennas
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