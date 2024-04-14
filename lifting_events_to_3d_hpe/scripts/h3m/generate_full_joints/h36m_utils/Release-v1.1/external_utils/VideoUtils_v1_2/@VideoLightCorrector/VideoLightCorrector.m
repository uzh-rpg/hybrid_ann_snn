classdef VideoLightCorrector < handle
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
    
    properties
        Correction
        Original
        Modified
        Function
        
        VP
        VPParams
        VideoName
        
        Frame
        FrameNum
        
        % Grau del polinomi que estimarem
        Grade = 16;
        
        Width
        Height
        
        Title = 'Video with Light Correction';
        Hima
        Hfig
        Haxes
    end
    
    methods
        function obj = VideoLightCorrector ( videoName, grade, varargin )
            obj.VideoName = videoName;
            obj.VPParams  = varargin;
            obj.Grade     = grade;
            
            obj.FrameNum = 1;
            
            try
                nameFile = [obj.VideoName(1:end - 4) '_LC' num2str(obj.Grade) '.mat'];
                eval(['load ' nameFile ';']);
                
                obj.Correction = correction;
                obj.Modified   = modified;
                obj.Original   = original;
                obj.Function   = f;
            catch e
                obj.processVideo();
            end
                
            obj.VP = VideoPlayer(videoName, varargin{:});
            
            obj.Width  = obj.VP.Width;
            obj.Height = obj.VP.Height; 
            
            obj.Frame = obj.VP.Frame * obj.Correction(obj.FrameNum);
            obj.Frame( obj.Frame > 1 ) = 1;
            obj.Frame( obj.Frame < 0 ) = 0;
        end
    end
end

