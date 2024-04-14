function varargout = transformImage(obj, ImageIn, T, varargin ) 
%   ProjectiveTransform: Class for applying projective transformations to
%   points and images.
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

    % ----------------------------
    % Checking Input Parameters
    % ----------------------------
    p = inputParser;   % Create instance of inputParser class.

    p.addRequired ( 'ImageIn', @check_Image );
    p.addRequired ( 'T', @check_T );
    
    p.addParamValue('Interpolation', 'linear', ...
        @(x)any(strcmpi(x,{'linear','cubic', 'nearest'})));
    
    p.addParamValue('Resampler', 'bound', ...
        @(x)any(strcmpi(x,{'symmetric','bound', 'replicate', 'circular'})));
    
    if ( size(ImageIn, 3) == 1 )
        p.addParamValue ( 'FillValues', 0, @check_Color );
    else
        p.addParamValue ( 'FillValues', [0 0 0], @check_Color );
    end
    
    p.addParamValue ( 'SameSize', false, @(x)x==false || x==true );
    
    p.parse(ImageIn, T, varargin{:});
    
    Interpolation = p.Results.Interpolation;
    FillValues    = p.Results.FillValues;
    SameSize      = p.Results.SameSize;
    Resampler     = p.Results.Resampler;
    % ----------------------------
    
    tform = maketform('projective', T);
    resamp = makeresampler({Interpolation,Interpolation},Resampler);

    CornersF = obj.getImageCorners ( ImageIn );
    
    newCornersGP = obj.transformPoints( CornersF, T );
    newIm = imtransform( ImageIn, tform, resamp, 'FillValues', FillValues','XYScale',1); 
    
    Tm = obj.getTm ( newCornersGP );
    
    Center = obj.transformPoints( [0 0], Tm );
%     hrec =obj.drawRectangle( newCornersGP, Tm, 'Crossed', true );
    if ( SameSize ) 
        [h w c] = size(ImageIn);
        [newH newW ~] = size(newIm);
        
        CenterO = round([w h] / 2);
        
        startXI = 1;
        endXI = w;
        
        startYI = 1;
        endYI = h;
        
        startXN = round(Center(1) - CenterO(1));
        endXN = round(Center(1) + CenterO(1)) - 1;
        
        startYN = round(Center(2) - CenterO(2));
        endYN = round(Center(2) + CenterO(2)) - 1;
        
        if ( startXN < 1 )
            startXI = 1 - startXN;
            startXN = 1;
        end
        
        if ( endXN > newW )
            endXN = newW;
        end
        
        endXI = startXI + (endXN - startXN);
        
        if ( startYN < 1 )
            startYI = 1 - startYN;
            startYN = 1;
        end
        
        if ( endYN > newH )
            endYN = newH;
        end
        
        endYI = startYI + (endYN - startYN);
        
        finalIma = zeros([h w c]);
        finalIma(startYI:endYI, startXI:endXI, :) = newIm(startYN:endYN, startXN:endXN, :);       
    else
        finalIma = newIm;
    end
    
    finalIma( finalIma > 1 ) = 1;
    finalIma( finalIma < 0 ) = 0;
    
    % Checking output parameters
    varargout{1} = finalIma;
    
    if ( nargout > 1 )
        varargout{2} = Center;
    end
    
    if ( nargout > 2 )
        varargout{3} = Tm;
    end
end

function res = check_Image ( Image )
    res = 1;
end

function res = check_T ( T ) 
    res = all(size(T) == [3 3]);
end

function res = check_Color ( Color ) 
    res = numel( Color ) == 3;
end