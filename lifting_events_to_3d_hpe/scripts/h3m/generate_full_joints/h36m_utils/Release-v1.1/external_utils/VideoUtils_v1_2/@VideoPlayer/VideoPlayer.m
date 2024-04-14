classdef  ...
        ( ...
          Hidden = false, ...          %If set to true, the class does not appear in the output of MATLAB commands or tools that display class names.
          InferiorClasses = {}, ...    %Use this attribute to establish a precedence relationship among classes. Specify a cell array of meta.class objects using the ? operator. The built-in classes double, single, char, logical, int64, uint64, int32, uint32, int16, uint16, int8, uint8, cell, struct, and function_handle are always inferior to user-defined classes and do not show up in this list.
          ConstructOnLoad = false, ... %If true, the class constructor is called automatically when loading an object from a MAT-file. Therefore, the construction must be implemented so that calling it with no arguments does not produce an error.
          Sealed = false ...           %If true, the class can be not be subclassed
         ) VideoPlayer < handle
% VideoPlayer   Class for read and play videos.
%
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 5 $
%   $Date: 2012-04-18 12:35:01 +0200 (Wed, 18 Apr 2012) $
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
%      Description
%      =============
%        Author: Marc Vivet - marc@vivet.cat
%        $Date: 2012-04-18 12:35:01 +0200 (Wed, 18 Apr 2012) $
%        $Revision: 5 $
%
%      Special Thanks
%      ===============
%        Oriol Martinez
%        Pol Cirujeda
%        Luis Ferraz 
%
%      Syntax:
%      ============
%
%        % Initialization
%        vp = VideoPlayer(videoName);
%        % Or
%        vp = VideoPlayer(videoName, 'PropertyName', PropertyValue, ...);
%        
%        % Next Frame
%        vp.nextFrame();
%        % Or 
%        vp + 1;
%
%        %Plots the Actual Frame
%        plot(vp);
%
%        % Delete the object
%        delete(vp);
%        % Or
%        clear vp;
%
%      Configurable Properties:
%      =========================
%        +--------------------+------------------------------------------+
%        | Property Name      | Description                              |
%        +====================+==========================================+
%        | Verbose            | Shows the internal state of the object   |
%        |                    | by generating messages.                  |
%        |                    | Is useful when debugging                 |
%        +--------------------+------------------------------------------+
%        | ShowTime           | Show the secods nedeed for each function |
%        |                    | of this classe. This paramter must be    |
%        |                    | combined with the Verbose mode.          |
%        +--------------------+------------------------------------------+
%        | InitialFrame       | Sets the initial frame of the video.     |
%        |                    | By default is 1. (It have to be an       |
%        |                    | integer).                                |
%        +--------------------+------------------------------------------+
%        | ImageSize          | Sets the returned image size.            |
%        |                    | The format is [width height].            |
%        +--------------------+------------------------------------------+
%        | StepInFrames       | Sets the number of skipped frames when we|
%        |                    | call the member function vp.NextFrame(). |
%        |                    | By default is 1. (It have to be an       |
%        |                    | integer).                                |
%        +--------------------+------------------------------------------+
%        | ValidRectangle     | Only shows the video region inside this  |
%        |                    | rectangle. (cuts the image) This option  |
%        |                    | is useful for cutting high resolution    |
%        |                    | videos.                                  |
%        +--------------------+------------------------------------------+
%        | UseStaticPicture   | Generates a video sequence using only an |
%        |                    | image. You have to specify the           |
%        |                    | transformation between each frame using  |
%        |                    | this format:                             |
%        |                    |    [shiftX shiftY Rotation Scalation]    |
%        +--------------------+------------------------------------------+
%        | MaxFrames          | Determines the max number of frames of   |
%        |                    | the current video.                       |
%        +--------------------+------------------------------------------+
%        | TransformOutput    | Apply a transformation to the output of  |
%        |                    | video frame using this format:           |
%        |                    |    [shiftX shiftY Rotation Scalation]    |
%        +--------------------+------------------------------------------+
%        | Binarize           | Binarizes the output image (true/false)  |
%        +--------------------+------------------------------------------+
%        | Title              | Let you set the figure title             |
%        +--------------------+------------------------------------------+
%        | UseSetOfVideos     | Let you open a ISV video (image set      |
%        |                    | video). You have to create this video by |
%        |                    | using the VideoRecorder class.           |
%        +--------------------+------------------------------------------+

    properties (GetAccess='public', SetAccess='public')
        CurrentPosition
    end

    properties (GetAccess='public', SetAccess='private')
        % Name of the video + path
        VideoName
        % Initial frame
        InitialFrame = 1;
        StepInFrames = 1;
        FrameNum
        Frame
        
        MaxFrames = 0;
        
        Width
        Height
        FrameWidth
        FrameHeight
        NumFrames
        Channels
        
        ValidRectangle = [0 0 0 0];
        
        ResizeImage = 0;
        ImageSize = [-1 -1];
        
        Hima
        Hfig
        Haxes
        
        SetVideoCurrentSet 
        SetVideoName  
        
        Tgp
    end
    
    properties (Hidden, Access = private)
        Id
        
        Quit = 0;
        
        CuttingImage = 0;
        
        TransformOutput = false;
        
        VideoPlayerIcons
        ToolButtonPlayStop
        ToolButtonQuick
        ToolButtonSlow
        ToolButtonStop
        ToolButtonAbout
        ToolButtonAntF
        ToolButtonNextF
        
        UseStaticPicture = 0;
        Increments
        MainFrame
        PMFCenter
        FrameCorners
        ImageCorners
        Tf
        Tfinv
        
        ImageCenter
        
        PT
        
        IsStaticPicture = 0;
        IsVideo = 0;
        
        IsSetOfVideos        = false;
        SetVideoMaxFrames
          
        SetVideoNumSets
        SetVideoFormat       
             
        SetVideoDir
        SetVideoFrameNum
        
        % Is a Sequence of images
        IsISV = 0;
        % Name of the sequence of images
        ISVName = '';
        % Extencion of these images
        ISVExt = '';
        
        % Show information in the cpp ffmpeg matlab library
        Verbose = false;
        % show process time of the cpp ffmpeg matlab library
        ShowTime = false;
        
        Title
        
        Binarize = false;
        BinarizeThreshold = 0.5;
       
               
    end
    
    methods
        % Constructor
        function obj = VideoPlayer(videoName, varargin)
            % ----------------------------
            % Checking Input Parameters
            % ----------------------------
            p = inputParser;   % Create instance of inputParser class.

            p.addRequired ( 'videoName' );
            
            p.addParamValue ( 'Verbose',            obj.Verbose,              @(x)x==false || x==true  );
            p.addParamValue ( 'ShowTime',           obj.ShowTime,             @(x)x==false || x==true  );
            p.addParamValue ( 'InitialFrame',       obj.InitialFrame,         @obj.check_param_InitialFrame );
            p.addParamValue ( 'ImageSize',          obj.ImageSize,            @obj.check_param_ImageSize );
            p.addParamValue ( 'MaxFrames',          obj.MaxFrames,            @obj.check_param_MaxFrames );
            p.addParamValue ( 'StepInFrames',       obj.StepInFrames,         @obj.check_param_StepInFrames );
            p.addParamValue ( 'UseStaticPicture',   obj.UseStaticPicture,     @obj.check_param_UseStaticPicture );
            p.addParamValue ( 'ValidRectangle',     obj.ValidRectangle,       @obj.check_param_ValidRectangle );
            p.addParamValue ( 'TransformOutput',    obj.TransformOutput,      @obj.check_param_TransformOutput );
            p.addParamValue ( 'Title',              videoName,                @isstr );
            p.addParamValue ( 'UseSetOfVideos',     obj.IsSetOfVideos,        @obj.check_param_UseSetOfVideos );
            p.addParamValue ( 'Binarize',           obj.Binarize,              @(x)x==false || x==true  );
            p.parse(videoName, varargin{:});
            
            obj.Verbose             = p.Results.Verbose;
            obj.ShowTime            = p.Results.ShowTime;
            obj.VideoName           = p.Results.videoName;
            obj.Title               = p.Results.Title;
            obj.Binarize            = p.Results.Binarize;
            % ----------------------------
         
                                               
            obj.FrameNum = obj.InitialFrame;
            
            if (exist(obj.VideoName) == 0)
                error(['File ''' videoName ''' do not exist!']);   
            end

            if (obj.UseStaticPicture) 
                % Reading Image
                
                obj.IsStaticPicture = 1;

                if (~obj.CuttingImage) 
                    error('In order to use static pictures you must define the ValidRectangle.');
                end

                frame = double(imread(obj.VideoName)) / 255.0;
                obj.MainFrame = frame;

                [hI wI ~] = size(frame);

                obj.PT = ProjectiveTransform ();

                [obj.Tf, obj.Tfinv] = obj.PT.getTf ( frame );

                obj.Tgp = eye(3);                    

                if (obj.ValidRectangle(1) == 0)
                    widthAux = obj.ValidRectangle(3);
                    heightAux = obj.ValidRectangle(4);                                           

                    offsetX = round((wI - widthAux) / 2.0) + 1;
                    offsetY = round((hI - heightAux) / 2.0) + 1;

                    obj.ValidRectangle = [offsetX offsetY round(widthAux) round(heightAux)];
                end

                obj.FrameWidth     = wI;
                obj.FrameHeight    = hI;

                %obj.NumFrames = Inf;
                
                if (obj.ResizeImage)
                    frame = imresize(frame, [obj.ImageSize(2) obj.ImageSize(1)]);
                    obj.Width = obj.ImageSize(1);
                    obj.Height = obj.ImageSize(2);
                    frame( frame > 1 ) = 1;
                    frame( frame < 0 ) = 0;
                end
            else                     
                % Opening Video
                
                if ( obj.IsSetOfVideos )
                    k = strfind(obj.VideoName, '/');
                    i = strfind(obj.VideoName, '_');
                    j = strfind(obj.VideoName, '.');
                    
                    obj.SetVideoCurrentSet = str2num(obj.VideoName(i(end) + 1:j(end) - 1));
                    
                    obj.SetVideoFormat = obj.VideoName(end-2:end);

                    if (numel(k) > 0)
                        obj.SetVideoName   = obj.VideoName(k(end)+1:i(end) - 1);
                        obj.SetVideoDir    = obj.VideoName(1:k(end));
                        
                    else
                        obj.SetVideoName = obj.VideoName(1:i(end) - 1);
                        obj.SetVideoDir  = '';
                    end                
                    
                                   
                        
                    if (obj.ResizeImage)                        
                        info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                            'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime, 'Scale', obj.ImageSize);
                    else
                        info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                            'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime);
                    end

                    obj.ResizeImage = 0;
                    
                    aux = dir([obj.SetVideoDir obj.SetVideoName '_*']);
                    obj.SetVideoNumSets = size(aux, 1);
                    
                    obj.Id                = info.Id;
                    obj.Width             = info.Width;
                    obj.Height            = info.Height;
                    obj.FrameWidth        = info.FrameWidth;
                    obj.FrameHeight       = info.FrameHeight;
                    obj.SetVideoMaxFrames = info.NumFrames;
                    
                    obj.NumFrames = obj.SetVideoNumSets * obj.SetVideoMaxFrames;

                    obj.SetVideoFrameNum = mod(obj.InitialFrame, obj.SetVideoMaxFrames); 
                    
                    currSet = floor(obj.InitialFrame / obj.SetVideoMaxFrames);
                    
                    if ( currSet ~= obj.SetVideoCurrentSet )
                       obj.SetVideoCurrentSet = currSet;
                       mexVideoReader(2, obj.Id);
                       
                       if (obj.ResizeImage)                        
                            info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                                'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime, 'Scale', obj.ImageSize);
                        else
                            info  = mexVideoReader(0, [obj.SetVideoDir obj.SetVideoName '_' num2str(obj.SetVideoCurrentSet) '.' obj.SetVideoFormat],...
                                'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime);
                        end

                        obj.Id                = info.Id;
                    end

                    if ( obj.SetVideoFrameNum ~= 1)
                        mexVideoReader(4, obj.Id, obj.SetVideoFrameNum);
                        frame = mexVideoReader(3, obj.Id);
                        obj.FrameNum = obj.InitialFrame + obj.SetVideoCurrentSet * obj.SetVideoMaxFrames;
                    else
                        frame = mexVideoReader(3, obj.Id);
                        obj.FrameNum = 1 + obj.SetVideoCurrentSet * obj.SetVideoMaxFrames;
                    end

                    if ( obj.TransformOutput )
                        obj.PT = ProjectiveTransform ();

                        [obj.Tf, obj.Tfinv] = obj.PT.getTf ( frame );

                        obj.Tgp = eye(3);
                    end
                else
                    if isdir(obj.VideoName)
                        obj.IsISV = 1;

                        k = strfind(obj.VideoName, '/');

                        if (numel(k) > 0)
                            obj.ISVName = obj.VideoName(k(end)+1:end-4);
                        else
                            obj.ISVName = obj.VideoName(1:end-4);
                        end

                        aux = dir([obj.VideoName '/*_*']);

                        obj.ISVExt = aux(end).name(end-3:end);
                        obj.NumFrames = size(aux, 1);

                        frame = obj.getISVFrame(obj.InitialFrame);

                        obj.FrameWidth  = size(frame, 2);
                        obj.FrameHeight = size(frame, 1);

                        if (obj.ResizeImage)
                            frame = imresize(frame, [obj.ImageSize(2) obj.ImageSize(1)]);
                            obj.Width = obj.ImageSize(1);
                            obj.Height = obj.ImageSize(2);
                            frame( frame > 1 ) = 1;
                            frame( frame < 0 ) = 0;
                        end     

                    else                    
                        obj.IsVideo = 1;                   

                        if (obj.ResizeImage)                        
                            info  = mexVideoReader(0, obj.VideoName,...
                                'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime, 'Scale', obj.ImageSize);
                        else
                            info  = mexVideoReader(0, obj.VideoName,...
                                'Verbose', obj.Verbose, 'ShowTime', obj.ShowTime);
                        end

                        obj.ResizeImage = 0;

                        obj.Id          = info.Id;
                        obj.Width       = info.Width;
                        obj.Height      = info.Height;
                        obj.FrameWidth  = info.FrameWidth;
                        obj.FrameHeight = info.FrameHeight;
                        obj.NumFrames   = info.NumFrames;

                       if ( obj.InitialFrame ~= 1)
                            mexVideoReader(4, obj.Id, obj.InitialFrame);
                            frame = mexVideoReader(3, obj.Id);
                            obj.FrameNum = obj.InitialFrame;
                       else
                            frame = mexVideoReader(3, obj.Id);
                       end

                       if ( obj.TransformOutput )
                            obj.PT = ProjectiveTransform ();

                            [obj.Tf, obj.Tfinv] = obj.PT.getTf ( frame );

                            obj.Tgp = eye(3);
                       end
                    end
                end
            end
             
            if (obj.CuttingImage)
                if ((obj.ValidRectangle(1) + obj.ValidRectangle(3)) > ...
                        obj.Width)

                    error('The Valid Rectangle does not fit to the video.');
                end

                if ((obj.ValidRectangle(2) + obj.ValidRectangle(4)) > ...
                        obj.Height)

                    error('The Valid Rectangle does not fit to the video.');
                end          

                obj.Width     = obj.ValidRectangle(3);
                obj.Height    = obj.ValidRectangle(4);                   
            end

            obj.Channels = size(frame, 3);

            obj.Frame = frame;
            if ( obj.Binarize )
                obj.Frame(obj.Frame >= obj.BinarizeThreshold) = 1;
                obj.Frame(obj.Frame <= obj.BinarizeThreshold) = 0;

            end
            
            if (obj.CuttingImage)
                obj.Frame = obj.cutImage(obj.Frame, obj.ValidRectangle);
            end

            obj.ImageCenter = [obj.Width obj.Height] / 2.0;
        end
        
        % Class Functions
        res = nextFrame(obj, varargin);
        
        % Destructor
        delete(obj); 
        
        res = plus (obj1, obj2);
        res = getFrameUInt8 ( obj );
        frame = getFrameAtNum ( obj, num );
        
        res = imShift( obj, im, offset );
        varargout = imTransform( obj, im, trans );
        subIm = cutImage (obj, ima, rec );
        
        setPlotPosition( obj, position );
        changeTransformIncrement (obj, value);
        setStepInFrames(obj, value);
        
        addGaussianNoise(obj, value);
        newPoints = getRelativePointPosition ( obj, points );
        
        frame = getISVFrame(obj, num);
        
        % Add a frame over the curren image, using a mask [r g b].
        ocluded = addFrameToFrame ( obj, frame, mask );
        
        % Visualization Functions
        disp(obj);
        plot(obj, varargin);
    end
       
    methods (Static, Hidden)
        % Auxiliary disp functions
        disp_Methods(obj);
        disp_HelpVideoPlayer(obj);
        disp_HelpNextFrame(obj);
        
        disp_HelpGetFrameUInt8(obj);
        disp_HelpGetFrameAtNum(obj);
        disp_HelpImShift(obj);
        disp_HelpImTransform(obj);
        disp_HelpCutImage(obj);
        disp_HelpAddGaussianNoise(obj);
        disp_HelpAddFrameToFrame(obj);
    end
    
    methods (Hidden, Access = public)
        % Check auxiliary function for param checking.
        check_param_InitialFrame(obj, value);        
        check_param_StepInFrames(obj, value);
        check_param_ValidRectangle(obj, value);
        check_param_MaxFrames(obj, value);
        check_param_UseStaticPicture(obj, value);
        check_param_ImageSize(obj, value);
        check_param_TransformOutput(obj, value);
    end
       
    methods (Access = private)       
        % Tool Bar functions
        tool_PauseVideo(obj);
        tool_ContinueVideo(obj);
        tool_AntFrame(obj);
        tool_NextFrame(obj);
        tool_QuickVideo(obj);
        tool_SlowVideo(obj);
        tool_About(obj);
        tool_Quit(obj);
    end
end
