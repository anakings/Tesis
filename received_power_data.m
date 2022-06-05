function [] = received_power_data(name, N, antenna_type, frequency, fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, SNR_list)
    % Compute antennas coordinates [N+1 x 3],
    % N+1: Number of antennas, 1 => Antenna at the center of the system
    % 3 => x, y, z
    antenna_coordinates = zeros(N+1, 3);
    for n = 1:N
        antenna_coordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_coordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_coordinates(n,3) = 0; % z axis                               
    end
    
    % Add antenna at the center of the receiving system
    antenna_coordinates(N+1,1) = 0;
    antenna_coordinates(N+1,2) = 0;
    antenna_coordinates(N+1,3) = r;
      
    % Compute power for each distance, antenna, azimuth angle, and
    % elevation angle
    for iter = 1:length(d_matrix)
        SNR_len_temp = size(SNR_list);
        SNR_len = SNR_len_temp(2);
        Pr = zeros(azimuthFinalAngle, elevationFinalAngle, N, SNR_len); % [360x180xN]
    
        d = d_matrix(iter); % Distance from source to center of the system
    
        for elevationAngle = 1:elevationFinalAngle
            for azimuthAngle = 1:azimuthFinalAngle
                % Compute the coordinates of the source
                x_t = d*cosd(azimuthAngle)*cosd(elevationAngle); % x axis
                y_t = d*sind(azimuthAngle)*cosd(elevationAngle); % y axis
                z_t = d*sind(elevationAngle); % z axis
                    
                % Get quadrant when the source is proyected on the xy plane
                if x_t < 0 && y_t < 0
                    qua = 3;
                elseif x_t < 0 && y_t >= 0
                    qua = 2;
                elseif y_t < 0
                    qua = 4;
                else
                    qua = 1;
                end
                
                for n = 1:N+1
                    % Get the coordinates of the receiving antennas
                    x_n = antenna_coordinates(n, 1);
                    y_n = antenna_coordinates(n, 2);
                    z_n = antenna_coordinates(n, 3);

                    % Compute the distance between the source and each of the antennas
                    d_n = sqrt(power(x_n-x_t, 2) + power(y_n-y_t, 2) + power(z_n-z_t, 2));

                    % Compute the elevation angle between the source and 
                    % each of the antennas [0:90]
                    elevation_n = asind((z_t - z_n)/d_n);
                    
                    % Compute the azimuth angle between the source and 
                    % each of the antennas
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
                    
                    % Readjust angles to be in the range [-180:180]
                    azimuth_n = azimuth_n - 180;
                    
                    % Create array with only one antenna
                    confarray = phased.ConformalArray('Element', antenna_type, 'ElementPosition', [x_n; y_n; z_n]);
                    
                    % Create signal
                    t = (0:1/fs:1).';
                    x1 = cos(2*pi*t*frequency); % Signal
                    doa = [azimuth_n; elevation_n]; % Signal direction
                    
                    % Received signal without noise
                    x = collectPlaneWave(confarray, x1, doa, frequency);
                    
                    for SNR = 1:SNR_len
                        % Add noise to received signal
                        variance = 1 / 10.^(SNR_list(SNR)/10); 
                        noise = variance*(randn(size(x)));
                        signal = x + noise;
                        
                        % Compute and save received signal power
                        Pr(azimuthAngle, elevationAngle, n, SNR) = rms(signal)^2;
                        
                        % Save distance between source and receiving antenna
                        Pr(azimuthAngle, elevationAngle, N+2, SNR) = d;
                        
                        % Save source proyected quadrant
                        Pr(azimuthAngle, elevationAngle, N+3, SNR) = qua;
                        
                        % Save SNR used to compute the noise added to the signal
                        Pr(azimuthAngle, elevationAngle, N+4, SNR) = SNR_list(SNR);
                    end
                end
            end
        end
        
        % Save matrix of power, distance, quadrant, and SNR
        fileName = strcat(name, int2str(iter));
        fileName = strcat(fileName, '.m');
        save(fileName, 'Pr');
    end
end
