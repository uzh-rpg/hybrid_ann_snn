classdef VideoPlayerGPyramid < handle
%VIDEOPLAYERGPYRAMID Reads a video and generates the gaussian pyramid.
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 14 $
%   $Date: 2012-04-24 18:27:21 +0200 (Tue, 24 Apr 2012) $
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
    
    properties(GetAccess='public', SetAccess='protected')
        Levels
        
        ItPSF = 2;
        PSF
               
        Vp
        Width
        Height
        Channels
        FrameNum
        
        Frame
        GPyramid
        
        Hima
        Hfig
        Haxes
    end
    
    methods
        function obj = VideoPlayerGPyramid ( videoName, levels, varargin )
            obj.Vp = VideoPlayer(videoName, varargin{:});
            
            obj.FrameNum = obj.Vp.FrameNum;
            obj.Levels = levels;
            obj.GPyramid = cell([obj.Levels 1]);
            
            a = [0.5 0.5];
            b = conv(a, a);

            for i = 1:obj.ItPSF
               b = conv(b, a);
            end

            obj.PSF = conv2(b, b');
            
            obj.Frame = obj.Vp.Frame;
            [obj.Height obj.Width obj.Channels] = size(obj.Frame);
            
            obj.framePyramid();
        end
    end
    
end

