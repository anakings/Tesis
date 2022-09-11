function [] = spatial_covariance_matrix_data_without_normalizate(name, N, antenna_type, fc, lambda, ...
    fs, r, d_matrix, azimuthFinalAngle, elevationFinalAngle, angle_step, ... 
    SNR_list, withLoss)

    % Compute antennas coordinates [N+1 x 3],
    % N+1: Number of antennas, 1 => Antenna at the center of the system
    % 3 => x, y, z
    antenna_coordinates = zeros(N, 3);
    
    for n = 1:N
        antenna_coordinates(n,1) = r*cosd(360*(n-1)/N); % x axis
        antenna_coordinates(n,2) = r*sind(360*(n-1)/N); % y axis
        antenna_coordinates(n,3) = 0; % z axis                               
    end
    
    act = antenna_coordinates.'; % act: Antenna coordinate transpose
    
    % Create receiving system
    confarray = phased.ConformalArray(...
        'Element', antenna_type, ...
        'ElementPosition', [act(1,:); act(2,:); act(3,:)]);
%     viewArray(confarray)
    
    % Create signal
    t = (0:1/fs:1).';
    x1 = cos(2*pi*t*fc); % Signal
      
    % Compute power for each distance, antenna, azimuth angle, and
    % elevation angle
    for iter = 1:length(d_matrix)
        SNR_len_temp = size(SNR_list);
        SNR_len = SNR_len_temp(2);
        
        % [360x180xN]
        lengthVector = 1:N-1;
        lengthVector = sum(lengthVector)*2;
        Pr = zeros(azimuthFinalAngle/angle_step, elevationFinalAngle/angle_step, lengthVector+3, SNR_len);
    
        d = d_matrix(iter); % Distance from source to center of the system
        
        azimuth_index = 0;
        for azimuthAngle = 1:angle_step:azimuthFinalAngle
            azimuth_index = azimuth_index + 1;
            elevation_index = 0;
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
            
            for elevationAngle = 1:angle_step:elevationFinalAngle
                elevation_index = elevation_index + 1;
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
                    
                    % Find the spatial covariance matrix of signal
                    covSignal = cov(signal);           %NxN
                    
                    % Since the matrix R is conjugate-symmetrical with 
                    %respect to the diagonal, the elements of its upper or 
                    %lower triangular part provide enough information for 
                    %2D DOA estimation.
                    covSignal = triu(covSignal,1);
                    
                    % Remove values = zero and convert the matrix to a vector
                    covSignalVector = nonzeros(covSignal(:))';
                    
                    % Divide into real and imaginary part
                    realPart = real(covSignalVector);
                    imagPart = imag(covSignalVector);
                    covSignalVector = [realPart imagPart];
              
                    Pr(azimuth_index, elevation_index, 1:lengthVector, SNR_index) = ...
                        covSignalVector;

                    % Save distance between source and center of the
                    % receiving system
                    Pr(azimuth_index, elevation_index, lengthVector+1, SNR_index) = angle_step;

                    % Save source proyected quadrant
                    Pr(azimuth_index, elevation_index, lengthVector+2, SNR_index) = qua;

                    % Save SNR used to compute the noise added to the
                    % signal
                    Pr(azimuth_index, elevation_index, lengthVector+3, SNR_index) = ...
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
