classdef MultibandBlending < handle
%MULTIBANDBLENDING Class to perform a Multi-Band Blending. See
%    P. BURT and E. ADELSON, A Multiresolution Spline with Application 
%    to Image Mosaics, Acm Transactions on Graphics, vol. 2, 
%    no. 4, pp. 217-236, 1983.
%
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 16 $
%   $Date: 2012-04-28 13:45:12 +0200 (Sat, 28 Apr 2012) $
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
%
%    Special Thanks
%    ===============
%      Elhanan Elboher 
%
%    Syntax:
%    ============
%      Initialization
%      mbb = MultibandBlending();
%      % Or
%      mbb = MultibandBlending('PropertyName', PropertyValue, ...);
%
%      % Stich two images
%      newImage = mbb.stitchLineal ( I1, I2 );
%      % Or
%      newImage = mbb.stitchMask ( I1, I2, binaryMask );
%      
%      % You can see an example by typing:
%      MultibandBlenging.example
%
%    Configurable Properties:
%    =========================
%      +--------------------+------------------------------------------+
%      | Property Name      | Description                              |
%      +====================+==========================================+
%      | NumLevels          | Number of Pyramid levels.                |
%      +--------------------+------------------------------------------+
%      | MaskBlur           | Number of iterations, used to blur the   |
%      |                    | the mask.                                |
%      +--------------------+------------------------------------------+
%      | ItPSF              | Number of iterations to generate the     |
%      |                    | gaussian kernel function. It must be     |
%      |                    | value.                                   |
%      +--------------------+------------------------------------------+
%      | ShowIntermediate   | Enables/Disables the creation of the     |
%      |                    | images in the different pyramid levels.  |
%      +--------------------+------------------------------------------+    
    properties
        % Pyramid Levels
        NumLevels = 3;
        
        % Struct containing all the Mask levels
        Mask
        % Blur iterations
        MaskBlur = 4;
        
        % Gaussian Kernel iterations ( should be Even)
        ItPSF = 2;
        
        %Gaussian Kernel
        PSF
        
        % Boolean that enables/disables the intermediate images
        ShowIntermediate = false;
    end
    
    methods
        % Class constructor
        function obj = MultibandBlending ( varargin )
            % ----------------------------
            % Checking Input Parameters
            % ----------------------------
            p = inputParser;   % Create instance of inputParser class.

            p.addParamValue ( 'NumLevels' , obj.NumLevels , ...
                @obj.check_param_NumLevels );
            p.addParamValue ( 'MaskBlur'  , obj.MaskBlur  , ...
                @obj.check_param_MaskBlur );
            p.addParamValue ( 'ItPSF'     , obj.ItPSF     , ...
                @obj.check_param_ItPSF );
            p.addParamValue ( 'ShowIntermediate', obj.ShowIntermediate, ...
                @(x)x==false || x==true );

            p.parse(varargin{:});
            
            obj.NumLevels        = p.Results.NumLevels;
            obj.MaskBlur         = p.Results.MaskBlur;
            obj.ItPSF            = p.Results.ItPSF;
            obj.ShowIntermediate = p.Results.ShowIntermediate;
            % ----------------------------
            
            % Generating the 2D Gaussian Kernel
            a = [0.5 0.5];
            b = conv(a, a);

            for i = 1:obj.ItPSF
               b = conv(b, a);
            end

            obj.PSF = conv2(b, b');
        end
    end
    
    methods (Access = public)
        % Stitch two images using the lineal mask ( half of the image )
        final = stitchLineal     ( obj, I1, I2 );
        % Stitch two images given a binary mask ( 0 , 1 ) values.
        final = stitchMask       ( obj, I1, I2, mask );
        % Stitch tow images given a binary mask (do not perform Multi-Band
        % blending.
        final = stitchMaskDirect ( obj, I1, I2, mask );
    end
    
    methods (Static)
       example ( obj ); 
    end
    
    methods (Access = private)
        data   = createGaussLaplaPyramid ( obj, I );
        maskFi = createLinealPyramidMask ( obj, w, h );
        maskFi = createMaskPyramidMask   ( obj, w, h, mask );
        result = stitch                  ( obj, data1, data2, dataMask);
    end
    
    methods (Hidden, Access = public)
        % Auxiliary function for parameters checking.
        check_param_NumLevels(obj, value); 
        check_param_MaskBlur(obj, value); 
        check_param_ItPSF(obj, value); 
    end   
end

