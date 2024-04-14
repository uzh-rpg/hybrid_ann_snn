function plot( obj )
%PLOT Shows the current image.
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

        
    if (numel(obj.Hima) == 0)
        obj.Hima  = zeros([obj.Levels 1]);
        obj.Hfig  = zeros([obj.Levels 1]);
        obj.Haxes = zeros([obj.Levels 1]);
        
        for i = 1:obj.Levels
                   
            obj.Hfig(i) = figure(...
                'numbertitle', 'off', ...
                'name', ['Level ' num2str(i) ' - Frame ' num2str(obj.FrameNum)], ...  
                'renderer', 'opengl', ...
                'Color', [0 0 0], ...
                'MenuBar',     'none');


            obj.Haxes(i) = axes( 'Parent',   obj.Hfig(i), ...
                              'Position', [0 0 1 1], ...
                              'DrawMode', 'fast');

            if ( i == 1)
                frame = obj.Frame;
            else
                frame = repmat(obj.GPyramid{i}.frame, [1 1 3]);
            end
            obj.Hima(i) = image(frame, 'Parent', ...
                obj.Haxes(i));

            set(obj.Haxes(i), 'Color', [0 0 0]);
            set(obj.Haxes(i), 'XTick', []);
            set(obj.Haxes(i), 'YTick', []);
            set(obj.Haxes(i), 'ZTick', []);
            set(obj.Haxes(i), 'Clipping', 'off');
        end
    end

    set(obj.Hima(1), 'CData', obj.Frame); 
    set(obj.Hfig(1), 'name' , ['Level 1 - Frame ' num2str(obj.FrameNum)]);
    
    for i = 2:obj.Levels
        frame = repmat(obj.GPyramid{i}.frame, [1 1 3]);
        
        set(obj.Hima(i), 'CData', frame); 
        set(obj.Hfig(i), 'name' , ['Level ' num2str(i) ' - Frame ' num2str(obj.FrameNum)]);
    end

end