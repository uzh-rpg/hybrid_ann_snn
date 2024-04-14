function oclusion = addFrameToFrame ( obj, frame, varargin ) 
% addFrameToFrame  overlap a frame to the current frame, and returns the
% percentage of ocluded pixels. The transparency is defined by the
% parameter <mask>, which it is an array of 3 values -> [r g b];
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
    

    % ----------------------------
    % Checking Input Parameters
    % ----------------------------
    p = inputParser;   % Create instance of inputParser class.

    p.addRequired ( 'frame' );

    p.addParamValue ( 'MaskColor',         [1 0 1], @(x)isnumeric(x) && numel(x)==3);
    p.addParamValue ( 'MultiBandBlending',   false, @(x)x==false || x==true  );
    p.addParamValue ( 'Mask',                   [], true); 
    p.addParamValue ( 'Levels',                  4, @(x)isnumeric(x) && x > 0);
    p.addParamValue ( 'MaskBlur',               12, @(x)isnumeric(x) && x > 0);
    p.addParamValue ( 'ItPSF',                   2, @(x)isnumeric(x) && x > 0);

    p.parse(frame, varargin{:});

    maskColor = p.Results.Verbose;
    mbb       = p.Results.MultiBandBlending;
    mask      = p.Results.Mask;
    levels    = p.Results.Levels;
    maskBlur  = p.Results.MaskBlur;
    itPSF     = p.Results.ItPSF;
    % ----------------------------

    [fh fw fc] = size(frame);
    [ch cw cc] = size(obj.Frame);
    
    if ( fh ~= ch || fw ~= cw || fc ~= cc )
        error('The input frame must have the same size of the video frame.');
    end
    
    if ( mbb )   
        mb = MultibandBlending('NumLevels', levels, 'MaskBlur', maskBlur, 'ItPSF', itPSF);
        
        if ( numel(mask) ~= 0)
            [mh mw mc] = size(mask);
         
            if ( fh ~= mh || fw ~= mw )
                error('The Muti Band Blending Mask must have the same size of the frame');
            end

            fMask = mask(:, :, 1);

            res = mb.stitchMaskDirect(obj.Frame, frame, fMask);
        else
            res = mb.stitchLineal(obj.Frame, frame);
        end 
        
        obj.Frame = res;
    else
        maskFrame = frame(:,:, 1) == maskColor(1) & frame(:, :, 2) == maskColor(2)  & ...
                        frame(:, :, 3) == maskColor(3);

        invMask = zeros(size(maskFrame));
        invMask( maskFrame == 0 ) = 1;

        numPixForeground = size(invMask(invMask == 1), 1);

        oclusion = (numPixForeground / (obj.Width * obj.Height)) * 100;

        obj.Frame = obj.Frame .* repmat(maskFrame, [1, 1, 3]) + ...
            repmat(invMask, [1, 1, 3]) .* frame;
    end
end