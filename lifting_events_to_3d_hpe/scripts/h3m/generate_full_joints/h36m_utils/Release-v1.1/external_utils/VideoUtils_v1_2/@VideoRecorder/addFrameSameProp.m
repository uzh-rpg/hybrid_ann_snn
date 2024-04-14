function finalVideo = addFrameSameProp( obj, frame ) 
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 2 $
%   $Date: 2012-04-16 22:44:46 +0200 (Mon, 16 Apr 2012) $
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are 
% met: 
%
% 1. Redistributions of source code must retain the above copyright notice, 
%    this list of conditions and the following disclaimer. 
% 2. Redistributions in binary form must reproduce the above copyright 
%    notice, this list of conditions and the following disclaimer in the 
%    documentation and/or other materials provided with the distribution. 
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER 
% OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% The views and conclusions contained in the software and documentation are
% those of the authors and should not be interpreted as representing 
% official policies, either expressed or implied, of the FreeBSD Project.

    if ( isinteger ( frame ) ) 
        frame = double ( frame ) / 255.0;
    end

    actSize = size(frame);
    
    myProp = obj.Size(1) / obj.Size(2);
    prop = actSize(2) / actSize(1);
    
    if ( myProp > prop )
        height = obj.Size(2);
        width = round(obj.Size(2) * prop);
    else       
        width = obj.Size(1);
        height = round(obj.Size(1) / prop);
    end

    newFrame = imresize(frame, [height width]);
 
    actSize = size(newFrame);
    
    offsetY = round((obj.Size(2) - actSize(1)) / 2);
    offsetX = round((obj.Size(1) - actSize(2)) / 2);
    
    finalVideo = zeros([obj.Size(2) obj.Size(1) 3]);

    finalVideo((1 + offsetY):(offsetY + actSize(1)), ...
        (1 + offsetX):(offsetX + actSize(2)), :) = newFrame;

    finalVideo(finalVideo < 0) = 0;
    finalVideo(finalVideo > 1) = 1;
    
    obj.addFrame(finalVideo);
end