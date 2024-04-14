function plot(obj)
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
        if (obj.IsSetOfVideos)
            set(obj.Hfig, 'name', [obj.SetVideoName ' - '  num2str(obj.FrameNum) '/' num2str(obj.SetVideoCurrentSet)]);
        else
            set(obj.Hfig, 'name', [obj.Title ' - '  num2str(obj.FrameNum)]);
        end
        
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

        load VideoPlayerIcons.mat;
        obj.VideoPlayerIcons = VideoPlayerIcons;
        clear VideoPlayerIcons;

        th = uitoolbar('Parent', obj.Hfig);


        obj.ToolButtonPlayStop = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.pause,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_PauseVideo(obj));

        obj.ToolButtonAntF = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.antFrame,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_AntFrame(obj));

        obj.ToolButtonNextF = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.nextFrame,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_NextFrame(obj));

        obj.ToolButtonSlow = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.slow,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_SlowVideo(obj));

        obj.ToolButtonQuick = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.quick,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_QuickVideo(obj));            

        obj.ToolButtonAbout = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.about,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_About(obj));                      

        obj.ToolButtonAbout = uipushtool('Parent', th, 'Cdata', obj.VideoPlayerIcons.exit,...
             'Enable', 'on', 'ClickedCallback', @(src,evnt)tool_Quit(obj));    
         
        if (obj.IsSetOfVideos)
            set(obj.Hfig, 'name', [obj.SetVideoName ' - '  num2str(obj.FrameNum) '/' num2str(obj.SetVideoCurrentSet)]);
        end
    end     
end