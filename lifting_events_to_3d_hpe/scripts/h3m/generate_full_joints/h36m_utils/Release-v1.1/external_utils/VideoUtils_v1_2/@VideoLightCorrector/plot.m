function plot(obj)
%VIDEOLIGTHCORRECTOR Class for reading videos and correction subit
%light changes, using a polynomial aproach.
%
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

    if (ishandle(obj.Hima))
        set(obj.Hfig, 'name', [obj.Title ' - '  num2str(obj.FrameNum)]);
        set(obj.Hima, 'CData', obj.Frame); 
    else
        obj.Hfig = figure(...
                        'numbertitle', 'off', ...
                        'name',        [obj.Title ' - '  num2str(obj.FrameNum)]);%, ...  
                        %'renderer',    'opengl');%, ...
                        %'MenuBar',     'none');

        obj.Haxes = axes( 'Parent',   obj.Hfig, ...
                    'Position', [0 0 1 1]);%, ...
                    %'DrawMode', 'fast');

        obj.Hima = image(obj.Frame, 'Parent', obj.Haxes);
        %obj.Hima = imshow(obj.Frame, 'Parent', obj.Haxes);

        set(obj.Haxes, 'XTick', []);
        set(obj.Haxes, 'YTick', []);
        set(obj.Haxes, 'ZTick', []);
        set(obj.Haxes, 'Clipping', 'off');

        axis(obj.Haxes, 'image');     

                     
    end       
end