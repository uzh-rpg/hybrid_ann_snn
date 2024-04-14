%% Simple VideoPlayer Example
% In this example we show how to use the *VideoPlayer* object.

%% Create a new VideoPlayer Object
% To generate a new *VideoPlayer* object we have to use the next sentence. 
% Note that it is not necessary to include the parameters Verbose and Showtime, these
% parameters are obtional.

% vp = VideoPlayer('./Resources/TestVideo.mp4', 'Verbose', false, 'ShowTime', true);
vp = VideoPlayer('/media/work/vicon/Irina/Videos/Waiting.54138969.mp4', 'Verbose', true, 'ShowTime', true,'StepInFrames',2);

%% Reproducing the video sequence
% Then we have to create a loop to play the entire video sequence:

while ( true )
   plot( vp );
   
   
   % Your code here.
   % To access to the current frame use -> vp.Frame
   
   
   drawnow;  
   if ( ~vp.nextFrame )
       break;
   end   
end

%% Releaseing the VideoPlayer Object
% After we have used the *VideoPlayer* object it is necessary to release it
% using this command:

clear vp;