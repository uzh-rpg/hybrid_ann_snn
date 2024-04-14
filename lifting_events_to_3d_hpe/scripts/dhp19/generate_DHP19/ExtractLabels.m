function [] = ExtractEventsToFramesAndMeanLabels(...
            fileID, ... % log file
            aedat, events, eventsPerFullFrame, ...
            startTime, stopTime, fileName, ...
            XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, ...
            xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ... % 1st mask coordinates
            xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, ... % 2nd mask coordinates
            do_subsampling, reshapex, reshapey, ...
            saveHDF5, convert_labels)
    
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
    img = zeros(sx*nbcam, sy);
    pose = zeros(13, 3);
    IMovie = NaN(nbcam, reshapex, reshapey, nbFrame_initialization);
    poseMovie = NaN(13, 3, nbFrame_initialization);

    last_k = 1;
    counter = 0;
    nbFrame = 0;
    

    countPerFrame = eventsPerFullFrame;

    
    %lastFrameTime = startTime;
    %lastTimeStampLastFrame = startTime; % initialization
    
    
    for idx = 1:length(timeStamp)

        % Constant event count accumulation.
        counter = counter + 1;

        if (counter >= countPerFrame)
            nbFrame = nbFrame + 1;
            % k is the time duration (in ms) of the recording up until the
            % current finished accumulated frame.
            k = floor((timeStamp(idx) - startTime)*0.0001)+1;
            
            % if k is larger than the label at the end of frame
            % accumulation, the generation of frames stops.
            if k > length(XYZPOS.XYZPOS.head)
                break;
            end
            
            %
            pose(1,:) = nanmean(XYZPOS.XYZPOS.head(last_k:k,:),1);
            pose(2,:) = nanmean(XYZPOS.XYZPOS.shoulderR(last_k:k,:),1);
            pose(3,:) = nanmean(XYZPOS.XYZPOS.shoulderL(last_k:k,:),1);
            pose(4,:) = nanmean(XYZPOS.XYZPOS.elbowR(last_k:k,:),1);
            pose(5,:) = nanmean(XYZPOS.XYZPOS.elbowL(last_k:k,:),1);
            pose(6,:) = nanmean(XYZPOS.XYZPOS.hipR(last_k:k,:),1);
            pose(7,:) = nanmean(XYZPOS.XYZPOS.hipL(last_k:k,:),1);
            pose(8,:) = nanmean(XYZPOS.XYZPOS.handR(last_k:k,:),1);
            pose(9,:) = nanmean(XYZPOS.XYZPOS.handL(last_k:k,:),1);
            pose(10,:) = nanmean(XYZPOS.XYZPOS.kneeR(last_k:k,:),1);
            pose(11,:) = nanmean(XYZPOS.XYZPOS.kneeL(last_k:k,:),1);
            pose(12,:) = nanmean(XYZPOS.XYZPOS.footR(last_k:k,:),1);
            pose(13,:) = nanmean(XYZPOS.XYZPOS.footL(last_k:k,:),1);
            
            poseMovie(:,:,nbFrame) = pose;
            
            last_k = k;
            %dt = timeStamp(idx) - lastTimeStampLastFrame;
            %lastTimeStampLastFrame = timeStamp(idx);
            
            % initialize for next frame.
            counter = 0;
            img = zeros(sx*nbcam,sy);
        end
    end
    
    disp(strcat('Number of frame: ',num2str(nbFrame)));
    fprintf(fileID, '%s \t frames: %d\n', fileName, nbFrame ); 
    
    if saveHDF5 == 1        
        if convert_labels == true
            Labelsfilenameh5 = strcat(fileName,'_label.h5');
            poseMovie = poseMovie(:,:,1:nbFrame);
        end
            if convert_labels == true
                h5create(Labelsfilenameh5,'/XYZ',[13 3 nbFrame])
                h5write(Labelsfilenameh5,'/XYZ',poseMovie)
            end
    end
end
