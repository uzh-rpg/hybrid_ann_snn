function res = nextFrame(obj, varargin)   
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 7 $
%   $Date: 2012-04-18 16:50:53 +0200 (Wed, 18 Apr 2012) $
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

    res = false;

    if (obj.MaxFrames ~= 0)
        if (obj.MaxFrames <= obj.FrameNum)
            obj.Quit = true;
        end
    end

    if (obj.Quit)     
        return;
    end

    if ( nargin == 2 )
        if (~isnumeric(varargin{1}))
            error('In order to add frames to a VideoPlayer, second argument must be an Integer.');
        elseif (varargin{1} ~= floor(varargin{1}))              
            error('In order to add frames to a VideoPlayer, second argument must be an Integer.');
        end

        stepNum = varargin{1};
    else
        stepNum = obj.StepInFrames;
    end

    FrameNum = obj.FrameNum + stepNum;

    if ((FrameNum > obj.NumFrames && obj.NumFrames > 0) || FrameNum <= 0) 
        return;
    end

    obj.FrameNum = FrameNum;
    
    if obj.IsVideo % Is a video file
        if ( mexVideoReader ( 4, obj.Id, obj.FrameNum ) )
            frame = mexVideoReader(3, obj.Id);
        else
            return;
        end            
    elseif obj.IsISV % Is an Image Set Video
        try
            frame  = obj.getISVFrame(obj.FrameNum); 
        catch e
            return;
        end
    elseif (obj.IsSetOfVideos)
        obj.SetVideoFrameNum = obj.SetVideoFrameNum + stepNum;
        
        if ( mexVideoReader ( 4, obj.Id, obj.SetVideoFrameNum ) )
            frame = mexVideoReader(3, obj.Id);
        else
            obj.SetVideoCurrentSet = obj.SetVideoCurrentSet + 1;
            
            obj.SetVideoFrameNum = mod(obj.SetVideoFrameNum, obj.SetVideoMaxFrames);
            
            if ( obj.SetVideoCurrentSet > obj.SetVideoNumSets - 1 )
                return;
            end
            
            mexVideoReader(2, obj.Id);
            
            if (obj.ResizeImage)                        
                info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                    'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime, 'Scale', obj.ImageSize);
            else
                info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                    'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime);
            end
            
            obj.Id                = info.Id;
            
            mexVideoReader ( 4, obj.Id, obj.SetVideoFrameNum );
            frame = mexVideoReader(3, obj.Id);

        end
    else% Is an Static Picture
        if (numel(obj.Increments) > 4)
            obj.CurrentPosition(1:5) = obj.CurrentPosition(1:5) + stepNum .* obj.Increments(1:5) ;
            obj.CurrentPosition(6) = obj.CurrentPosition(6) * obj.Increments(6).^stepNum;
        else
            obj.CurrentPosition(1:3) = obj.CurrentPosition(1:3) + stepNum .* obj.Increments(1:3) ;
            obj.CurrentPosition(4) = obj.CurrentPosition(4) * obj.Increments(4).^stepNum;
        end

        frame = obj.imTransform(obj.MainFrame, obj.CurrentPosition);    
    end
    
    if (obj.ResizeImage)
       frame = imresize(frame, [obj.ImageSize(2) obj.ImageSize(1)]);
       frame( frame > 1 ) = 1;
       frame( frame < 0 ) = 0;
    end
    
    if (obj.CuttingImage)
        frame = obj.cutImage(frame, obj.ValidRectangle);
    end  
    
    if ( obj.TransformOutput )
        if (numel(obj.Increments) > 4)
            obj.CurrentPosition(1:5) = obj.CurrentPosition(1:5) + stepNum .* obj.Increments(1:5) ;
            obj.CurrentPosition(6) = obj.CurrentPosition(6) * obj.Increments(6).^stepNum;
        else
            obj.CurrentPosition(1:3) = obj.CurrentPosition(1:3) + stepNum .* obj.Increments(1:3) ;
            obj.CurrentPosition(4) = obj.CurrentPosition(4) * obj.Increments(4).^stepNum;
        end
      
        frame = obj.imTransform(frame, obj.CurrentPosition); 
    end
    
    obj.Frame = frame;
    if ( obj.Binarize )
        obj.Frame(obj.Frame >= obj.BinarizeThreshold) = 1;
        obj.Frame(obj.Frame <= obj.BinarizeThreshold) = 0;       
    end

    res = true;
end