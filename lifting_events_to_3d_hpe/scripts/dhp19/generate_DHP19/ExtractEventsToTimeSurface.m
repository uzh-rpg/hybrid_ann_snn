function [] = ExtractEventsToVoxelAndMeanLabels(...
    fileID, ... % log file
    aedat, events, eventsPerFullFrame, ...
    startTime, stopTime, fileName, ...
    XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, ...
    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ... % 1st mask coordinates
    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, ... % 2nd mask coordinates
    do_subsampling, reshapex, reshapey, ...
    saveHDF5, convert_labels)



save_count_frames = false;
startTime = uint32(startTime);
stopTime  = uint32(stopTime);

% Extract and filter events from aedat
[startIndex, stopIndex, pol, X, y, cam, timeStamp] = ...
    extract_from_aedat(...
    aedat, events, ...
    startTime, stopTime, ...
    sx, sy, nbcam, ...
    thrEventHotPixel, dt, ...
    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ...
    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2);

% Initialization
nbFrame_initialization = round(length(timeStamp)/eventsPerFullFrame);
acc = zeros(sx*nbcam, sy, 2);


save_output= true;
counter = 0;
nbFrame = 1;
delta = 300000;

countPerFrame = eventsPerFullFrame;


%lastFrameTime = startTime;
%lastTimeStampLastFrame = startTime; % initialization

init_slice = 1;

t0 = timeStamp(init_slice);
for idx = 1:length(timeStamp)
    
    coordx = X(idx);
    coordy = y(idx);
    pi = pol(idx);
    ti = timeStamp(idx);
    acc(coordx, coordy, pi) = ti;
    
    % Constant event count accumulation.
    counter = counter + 1;
    
    if counter >= eventsPerFullFrame
        t0 = timeStamp(idx);
        img = exp(-(double(t0) - acc) / delta);
        
        
        % k is the time duration (in ms) of the recording up until the
        % current finished accumulated frame.
        k = floor((timeStamp(idx) - startTime)*0.0001)+1;        
        % if k is larger than the label at the end of frame
        % accumulation, the generation of frames stops.
        if k > length(XYZPOS.XYZPOS.head)
            break;
        end
        
        % arrange image in channels.
        I1=img(1:sx,:, :);
        I2=img(sx+1:2*sx,:, :);
        I3=img(2*sx+1:3*sx,:, :);
        I4=img(3*sx+1:4*sx,:, :);      
             
        % Normalization
        V1n = uint8(normalizeImage3Sigma(I1(:, :, 1) - I1(:, :, 2)));
        V2n = uint8(normalizeImage3Sigma(I2(:, :, 1) - I2(:, :, 2)));
        V3n = uint8(normalizeImage3Sigma(I3(:, :, 1) - I3(:, :, 2)));
        V4n = uint8(normalizeImage3Sigma(I4(:, :, 1) - I4(:, :, 2)));

        if save_output
             DVSfilename = strcat(fileName, '_frame_', string(nbFrame-1), '_cam_0_', 'timesurface.mat');
             save(DVSfilename, 'V1n')
             DVSfilename = strcat(fileName, '_frame_', string(nbFrame-1), '_cam_1_', 'timesurface.mat');
             save(DVSfilename, 'V2n')
             DVSfilename = strcat(fileName, '_frame_', string(nbFrame-1), '_cam_3_', 'timesurface.mat');
             save(DVSfilename, 'V4n')
             DVSfilename = strcat(fileName, '_frame_', string(nbFrame-1), '_cam_2_', 'timesurface.mat');
             save(DVSfilename, 'V3n')
        end
        nbFrame = nbFrame+1;
        
        % nbFrame = nbFrame+1;initialize for next frame.
        counter = 0;

    end
end

disp(strcat('Number of frame: ',num2str(nbFrame)));
fprintf(fileID, '%s \t frames: %d\n', fileName, nbFrame );

end
