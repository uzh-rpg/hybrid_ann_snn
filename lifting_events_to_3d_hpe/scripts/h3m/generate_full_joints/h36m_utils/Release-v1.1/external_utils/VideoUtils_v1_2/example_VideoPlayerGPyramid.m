%% Simple VideoPlayerGPyramid Example
% In here you can see and example of how to use the *VideoPlayerGPyramid* 
% object in order to reproduce a Video as a gaussian pyaramid.

%% Create a new VideoPlayerGPyramid Object
% To generate a new *VideoPlayerGPyramid* object we have to use the next 
% sentence, where 'levels' is the number of gaussian pyramid levels. 

levels = 4; % Number of gaussian pyramid levels

vpgp = VideoPlayerGPyramid('./Resources/TestVideo.mp4', levels);

%% Define the position of the windows
% This part is optional but if you want to order the windows (one for each
% level of the gaussian pyramid) the you must do as follows:

startX = 100;
startY = 600;
width  = 466;
height = 350;

vpgp.setPosition ( startX, startY, width, height);

%% Play the video sequence
% Then you need to include this loop in order to play the entire video
% sequence:

while ( true )
   plot( vpgp );
   
   drawnow;
   if ( ~vpgp.nextFrame )
       break;
   end   
end

%% Release the VideoPlayerGPyarmid Object
% Finally you have to release the object.

clear vpgp;