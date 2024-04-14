%% VideoPlayer Example using an static picture
% The *VideoPlayer* object is capable of creating an online video using an
% static picture

%% Create the VideoPlayer object
% In order to create a *VideoPlayer* object using an static picture we have 
% to define the *VideoPlayer* object using the optional parameter 
% 'UseStaticPicture', where we have define a projective transform, which 
% will be applied to the image for each frame. See next source code:

pictureName = 'peppers.png'; % Static picture

transX = 10; % X Translation
transY = 5;  % Y Translation

rotX = 0;    % Rotation in the X axis (in degrees)
rotY = 0;    % Rotation in the Y axis (in degrees)
rotZ = 10;   % Rotation in the Z axis (in degrees)
    
scale = 1.08; % Scale factor

imageSize = [320, 240]; % Cutted region of the static picture

numberFrames = 10;      % Number of frames of the synthetic video.

vp = VideoPlayer(pictureName, ...
    'UseStaticPicture', [transX transY rotX rotY rotZ scale], ...
    'ValidRectangle', imageSize, 'MaxFrames', numberFrames);

%% Reproduce the synthetic video
% In order to reproduce the synthetic video you have to create the loop
% like in the example_VideoPlayer.m.

while(true)   
    plot(vp); 
    
    disp( mat2str(vp.Tgp) );
    
    drawnow;
    
    if (~vp.nextFrame)
        break;
    end
end


%% Release the VideoPlayer object
% Finally you have to release the object.

clear vp;