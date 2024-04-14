classdef  ...
        ( ...
          Hidden = false, ...          %If set to true, the class does not appear in the output of MATLAB commands or tools that display class names.
          InferiorClasses = {}, ...    %Use this attribute to establish a precedence relationship among classes. Specify a cell array of meta.class objects using the ? operator. The built-in classes double, single, char, logical, int64, uint64, int32, uint32, int16, uint16, int8, uint8, cell, struct, and function_handle are always inferior to user-defined classes and do not show up in this list.
          ConstructOnLoad = false, ... %If true, the class constructor is called automatically when loading an object from a MAT-file. Therefore, the construction must be implemented so that calling it with no arguments does not produce an error.
          Sealed = false ...           %If true, the class can be not be subclassed
         ) VideoRecorder < handle
% VideoRecorder   Class for record videos.
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
%

    properties (GetAccess='public', SetAccess='private')   
        % Video Name
        FileName
        % Video Codec
        VideoFormat
        % Frames per second
        Fps = 25;
        % Bits per second ( Hight values means hight quality )
        Bps = 8000000;
        % Video frame size
        Size = [720 512];
        % Number of Frames of the current video
        NumFrames = 0;
    end

    properties (Hidden, Access = 'private')
        % 
       % FirstTime = 1;
        % mexVideoWriter identifier
        Id
        % Format extension
        Format = 'mov'
        % Video name withoud extension
        VideoName
        % Video path
        VideoPath = '';
        % Max number of frames for a video sequence is used to split the
        % video in diferent parts.
        MaxFrames = 0;
        % Part counter ( used with MaxFrames )
        Part = 0;

        % Show information in the cpp ffmpeg matlab library
        Verbose = false;
        % show process time of the cpp ffmpeg matlab library
        ShowTime = false;
        
        % Requiered information for Image Set Videos (ISV) video
        % --------------------------------------------------------
        % Determines that this video is a Image Set Video
        IsISV = false;
        % Folder where the images are stored
        VideoFolder = '';
        % Image format.
        ImageFormat = 'png';       
    end
    
    methods
        % Constructor
        function obj = VideoRecorder(videoName, varargin)
            % ----------------------------
            % Checking Input Parameters
            % ----------------------------
            p = inputParser;   % Create instance of inputParser class.

            p.addRequired ( 'videoName' );
            
            p.addParamValue ( 'Verbose',   obj.Verbose,   @(x)x==false || x==true  );
            p.addParamValue ( 'ShowTime',  obj.ShowTime,  @(x)x==false || x==true  );
            p.addParamValue ( 'Fps',       obj.Fps,       @obj.check_param_Fps );
            p.addParamValue ( 'Bps',       obj.Bps,       @obj.check_param_Bps );
            p.addParamValue ( 'Format',    obj.Format,    @obj.check_param_Format );
            p.addParamValue ( 'Size',      obj.Size,      @obj.check_param_Size );
            p.addParamValue ( 'MaxFrames', obj.MaxFrames, @obj.check_param_MaxFrames );

            p.parse(videoName, varargin{:});
            
            obj.Verbose   = p.Results.Verbose;
            obj.ShowTime  = p.Results.ShowTime;
            obj.VideoName = p.Results.videoName;
            obj.Fps       = p.Results.Fps;
            obj.Bps       = p.Results.Bps;
            obj.Format    = p.Results.Format;
            obj.Size      = p.Results.Size;
            obj.MaxFrames = p.Results.MaxFrames;
            % ----------------------------
     
            k = strfind(videoName, '/');
                
            if (numel(k) > 0)
                obj.VideoName = videoName(k(end)+1:end);
                obj.VideoPath = videoName(1:k(end));
            else
                obj.VideoName = videoName;
                obj.VideoPath = [pwd '/'];
            end
            
            if ( obj.IsISV ) % Image Set Video
                obj.FileName = [obj.VideoName '.isv'];
                
                obj.VideoFormat = 'ISV Image Set Video';
                
                obj.VideoFolder = [obj.VideoPath obj.VideoName '.isv'];
                
                obj.Fps = 'Not Used';
                obj.Bps = 'Not Used';
                
                if (isdir(obj.VideoFolder))
                    strResponse = input([obj.VideoFolder ' already exist, do you want to overwrite? ([y] / n)'], 's');
                    if isempty(strResponse)
                        strResponse = 'y';
                    end
                    
                    if (strResponse == 'y')
                        rmdir(obj.VideoFolder, 's');
                    else
                        error('Video Name already exist.');
                    end
                end
                
                mkdir(obj.VideoFolder);
            else     
                if ( obj.MaxFrames )
                    videoName = [obj.VideoName '_' num2str(obj.Part)];
                    obj.FileName = [videoName '.' obj.Format];
                else
                    videoName = obj.VideoName;
                    obj.FileName = [obj.VideoName '.' obj.Format];
                end
                
                info = mexVideoWriter(0, [obj.VideoPath videoName], 'Format', obj.Format, 'Size', obj.Size, 'Fps', obj.Fps, 'Bps', obj.Bps, 'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime);
                obj.Id = info.Id;
                obj.VideoFormat = info.Format;
            end
        end
        
        % Class Functions
        addFrame(obj, frame);
        addFrame3Channels(obj, frame);
        finalVideo = addFrameSameProp( obj, frame );
        
        resFi = addFrame4Window(obj, frame1, frame2, frame3, frame4);
        
        frameAux = convert2ThreeChannels(obj, frame);
        
        finalVideo = resizeSameProp( obj, frame, newSize );
        finalVideo = overlapWindow( obj, oriframe, frame, position );
        
        frame = getFigureFrame ( obj, hfig );
        
        % Destructor
        delete(obj); 
    end
    
    methods (Hidden, Access = 'private')
        check_MaxFrames ( obj )
    end
    
    methods (Static, Hidden)
        % Auxiliary disp functions
    end
    
    methods (Hidden, Access = 'public')
        % Check auxiliary function for param checking.
        check_param_Fps(obj, value);        
    end
end
