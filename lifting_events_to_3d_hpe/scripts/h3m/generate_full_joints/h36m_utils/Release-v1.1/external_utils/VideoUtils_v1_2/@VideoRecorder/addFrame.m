function addFrame(obj, frame)
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

    obj.NumFrames = obj.NumFrames + 1;     

    if ( ~obj.IsISV )
        mexVideoWriter(1, obj.Id, frame);         
        obj.check_MaxFrames ();         
    else       
        numFrame = num2str(obj.NumFrames);
        if (obj.NumFrames < 10)
            numFrame = ['00000' numFrame];
        else
            if (obj.NumFrames < 100)
                numFrame = ['0000' numFrame];
            else
                if (obj.NumFrames < 1000)
                    numFrame = ['000' numFrame];
                else
                    if (obj.NumFrames < 10000)
                        numFrame = ['00' numFrame];
                    else
                        if (obj.NumFrames < 100000)
                            numFrame = ['0' numFrame];
                        end
                    end
                end
            end
        end
        
        imwrite(imresize(frame, [obj.Size(2) obj.Size(1)]), ...
            [obj.VideoFolder  '/' obj.VideoName '_' numFrame '.' ...
            obj.ImageFormat], obj.ImageFormat);        
    end    
end