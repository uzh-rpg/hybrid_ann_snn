function [] = ExtractEventsToVoxel(...
    fileID, ... % log file
    aedat, events, eventsPerFullFrame, ...
    startTime, stopTime, fileName, ...
    XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, ...
    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ... % 1st mask coordinates
    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, ... % 2nd mask coordinates
    do_subsampling, reshapex, reshapey, ...
    saveHDF5, convert_labels)

voxel_filenameh5 = strcat(fileName,'_voxel.h5');
if exist(voxel_filenameh5, 'file') == 2
    return
end


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
B = 4;
nbFrame_initialization = round(length(timeStamp)/eventsPerFullFrame);
img = zeros(sx*nbcam, sy);
voxel = zeros(sx*nbcam, sy, B);
pose = zeros(13, 3);


IMovie = NaN(nbcam, reshapex, reshapey, nbFrame_initialization);
VoxelMovie = NaN(nbcam, reshapex, reshapey, B, nbFrame_initialization);

poseMovie = NaN(13, 3, nbFrame_initialization);

last_k = 1;
counter = 0;
nbFrame = 1;


countPerFrame = eventsPerFullFrame;


%lastFrameTime = startTime;
%lastTimeStampLastFrame = startTime; % initialization

init_slice = 1;

t0 = timeStamp(init_slice);
dt = double(timeStamp(init_slice+eventsPerFullFrame) - t0);

for idx = 1:length(timeStamp)
    
    coordx = X(idx);
    coordy = y(idx);
    pi = pol(idx);
    ti = timeStamp(idx);
    
    % Constant event count accumulation.
    counter = counter + 1;
    img(coordx,coordy, 1) = img(coordx,coordy, 1) + 1;
    t = double(B -1 ) / dt * double(ti - t0) + 1;
    for tn=1:B
        voxel(coordx,coordy, tn) = voxel(coordx,coordy, tn) +  pi * max(0, 1 - abs(tn - t));
    end
    
    
    if (counter >= countPerFrame)

        init_slice = idx+1;
        final_slice = min(init_slice+eventsPerFullFrame, length(timeStamp));
        t0 = timeStamp(init_slice);
        dt = double(timeStamp(final_slice) - t0);
        
        % k is the time duration (in ms) of the recording up until the
        % current finished accumulated frame.
        k = floor((timeStamp(idx) - startTime)*0.0001)+1;
        
        % if k is larger than the label at the end of frame
        % accumulation, the generation of frames stops.
        if k > length(XYZPOS.XYZPOS.head)
            break;
        end
        
        % arrange image in channels.
        I1=img(1:sx,:);
        I2=img(sx+1:2*sx,:);
        I3=img(2*sx+1:3*sx,:);
        I4=img(3*sx+1:4*sx,:);
        
        % arrange image in channels.
        v1=voxel(1:sx,:, :);
        v2=voxel(sx+1:2*sx,:, :);
        v3=voxel(2*sx+1:3*sx,:, :);
        v4=voxel(3*sx+1:4*sx,:, :);
        
        
        
        % subsampling
        if do_subsampling
            I1s = subsample(I1,sx,sy,reshapex,reshapey, 'center');
            % different crop location as data is shifted to right side.
            I2s = subsample(I2,sx,sy,reshapex,reshapey, 'begin');
            I3s = subsample(I3,sx,sy,reshapex,reshapey, 'center');
            I4s = subsample(I4,sx,sy,reshapex,reshapey, 'center');
            
            v1 = subsample(v1,sx,sy,reshapex,reshapey, 'center');
            % different crop location as data is shifted to right side.
            v2 = subsample(v2,sx,sy,reshapex,reshapey, 'begin');
            v3 = subsample(v3,sx,sy,reshapex,reshapey, 'center');
            v4 = subsample(v4,sx,sy,reshapex,reshapey, 'center');
        else
            I1s = I1;
            I2s = I2;
            I3s = I3;
            I4s = I4;
        end
        
        % Normalization
        I1n = uint8(normalizeImage3Sigma(I1s));
        I2n = uint8(normalizeImage3Sigma(I2s));
        I3n = uint8(normalizeImage3Sigma(I3s));
        I4n = uint8(normalizeImage3Sigma(I4s));
        
        V1n = uint8(v1);
        V2n = uint8(v2);
        V3n = uint8(v3);
        V4n = uint8(v4);
        
        
%         
%          DVSfilename = strcat(fileName, '_', string(nbFrame), '.mat');
%          save(DVSfilename, 'V3n')
         DVSfilename = strcat(fileName, '_', string(nbFrame-1), '_cam_0_', '.mat');
         save(DVSfilename, 'V1n')
                  DVSfilename = strcat(fileName, '_', string(nbFrame-1), '_cam_1_', '.mat');
         save(DVSfilename, 'V2n')
                  DVSfilename = strcat(fileName, '_', string(nbFrame-1), '_cam_3_', '.mat');
         save(DVSfilename, 'V4n')
                           DVSfilename = strcat(fileName, '_', string(nbFrame-1), '_cam_2_', '.mat');
         save(DVSfilename, 'V3n')
        
        VoxelMovie(1,:,:,:,nbFrame) = V1n;
        VoxelMovie(2,:,:,:,nbFrame) = V2n;
        VoxelMovie(3,:,:,:,nbFrame) = V3n;
        VoxelMovie(4,:,:,:,nbFrame) = V4n;
        
        last_k = k;
        %dt = timeStamp(idx) - lastTimeStampLastFrame;
        %lastTimeStampLastFrame = timeStamp(idx);
        
        % initialize for next frame.
        counter = 0;
        img = zeros(sx*nbcam,sy);
        voxel = zeros(sx*nbcam, sy, B);
                nbFrame = nbFrame + 1;  
    end
end

disp(strcat('Number of frame: ',num2str(nbFrame)));
fprintf(fileID, '%s \t frames: %d\n', fileName, nbFrame );

end
