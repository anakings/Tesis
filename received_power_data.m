function [] = received_power_data(name, N, antenna_type, fc, lambda, ...
    fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, SNR_list, ...
    withLoss)

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
    
    act = antenna_coordinates.'; % act: Antenna coordinate transpose
    
    % Create receiving system
    confarray = phased.ConformalArray(...
        'Element', antenna_type, ...
        'ElementPosition', [act(1,:); act(2,:); act(3,:)]);
%     viewArray(confarray)
            
    % Array receiving frequency range
%     minimumFreq = fc - 10e6;
%     maximumFreq = fc + 10e6;
%     confarray.Element.FrequencyRange = [minimumFreq maximumFreq];
    
    % Create signal
    t = (0:1/fs:1).';
    x1 = cos(2*pi*t*fc); % Signal
      
    % Compute power for each distance, antenna, azimuth angle, and
    % elevation angle
    for iter = 1:length(d_matrix)
        SNR_len_temp = size(SNR_list);
        SNR_len = SNR_len_temp(2);
        
        % [360x180xN]
        Pr = zeros(azimuthFinalAngle, elevationFinalAngle, N, SNR_len);
    
        d = d_matrix(iter); % Distance from source to center of the system
        
        for azimuthAngle = 1:azimuthFinalAngle
            % Get quadrant when the source is proyected on the xy plane
            if azimuthAngle < 90
                qua = 1;
            elseif azimuthAngle >= 90 && azimuthAngle < 180
                qua = 2;
            elseif azimuthAngle >= 180 && azimuthAngle < 270
                qua = 3;
            else
                qua = 4;
            end
            
            for elevationAngle = 1:elevationFinalAngle         
                % Compute the coordinates of the source
                x_t = d*cosd(azimuthAngle)*cosd(elevationAngle); % x axis
                y_t = d*sind(azimuthAngle)*cosd(elevationAngle); % y axis
                z_t = d*sind(elevationAngle); % z axis
                
                % Readjust angles to be in the range [-180:180]
                azimuthAngleAdj = azimuthAngle - 180;
                    
                doa = [azimuthAngleAdj; elevationAngle]; % Signal direction

                % Received signal without noise
                x = collectPlaneWave(confarray, x1, doa, fc);

                for SNR_index = 1:SNR_len
                    % Add noise to received signal
                    variance = 1 / 10.^(SNR_list(SNR_index)/10); 
                    noise = sqrt(variance) * randn(size(x));
                    signal = x + noise;
                    
                    for n = 1:N+1
                        if withLoss
                            % Get the coordinates of the receiving antennas
                            x_n = antenna_coordinates(n, 1);
                            y_n = antenna_coordinates(n, 2);
                            z_n = antenna_coordinates(n, 3);

                            % Compute the distance between the source and
                            % each of the antennas
                            d_n = sqrt(power(x_n-x_t, 2) + ...
                                power(y_n-y_t, 2) + ...
                                power(z_n-z_t, 2));

                            % propagation loss (dB)
                            LdB = fspl(d_n, lambda);
                            
                            L = 10^(LdB/10);
                        else
                            L = 1;
                        end
                        
                        % Compute and save received signal power
                        Pr(...
                            azimuthAngle, ...
                            elevationAngle, ...
                            n, ...
                            SNR_index) = mean(abs(signal(:, n)).^2) / L^2;
                    end

                    % Save distance between source and center of the
                    % receiving system
                    Pr(azimuthAngle, elevationAngle, N+2, SNR_index) = d;

                    % Save source proyected quadrant
                    Pr(azimuthAngle, elevationAngle, N+3, SNR_index) = qua;

                    % Save SNR used to compute the noise added to the
                    % signal
                    Pr(azimuthAngle, elevationAngle, N+4, SNR_index) = ...
                        SNR_list(SNR_index);
                end
            end
        end
        
        % Save matrix of power, distance, quadrant, and SNR
        fileName = strcat(name, int2str(iter));
        fileName = strcat(fileName, '.m');
        save(fileName, 'Pr');
    end
end
