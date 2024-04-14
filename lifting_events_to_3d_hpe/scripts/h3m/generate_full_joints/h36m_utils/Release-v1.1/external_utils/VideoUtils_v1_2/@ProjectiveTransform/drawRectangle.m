function rec = drawRectangle ( obj, Corners, T, varargin )
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

    p = inputParser;   % Create instance of inputParser class.

    p.addRequired ( 'Corners', @check_Corners );
    p.addRequired ( 'T', @check_T );
    
    p.addParamValue ( 'Color', [1 0 0], @check_Color );
    p.addParamValue ( 'LineWidth', 2, @isnumeric );
    p.addParamValue ( 'LineStyle', '-', @ischar );
    p.addParamValue ( 'Haxes', gca, @ishandle );   
    p.addParamValue ( 'Crossed', false, @(x)x==false || x==true );
    
    p.addOptional ( 'Ignore', [], @(x)x==0);
    
    p.parse(Corners, T, varargin{:});
    
    Color = p.Results.Color;
    LineWidth = p.Results.LineWidth;
    LineStyle = p.Results.LineStyle;
    Haxes = p.Results.Haxes;
    Crossed = p.Results.Crossed;

    newP = obj.transformPoints ( Corners, T );
    
    if ( Crossed )
        rec = zeros(6, 1);
        
        rec(5) = line([newP(1,1) newP(3,1)], [newP(1,2) newP(3,2)], 'Color',...
            Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
            'Parent', Haxes);
        
        rec(6) = line([newP(2, 1) newP(4, 1)], [newP(2, 2) newP(4,2)], 'Color',...
            Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
            'Parent', Haxes);
    else
        rec = zeros(4, 1);
    end
    
    rec(1) = line([newP(1,1) newP(2,1)], [newP(1,2) newP(2,2)], 'Color',... 
        Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
        'Parent', Haxes);
    rec(2) = line([newP(2,1) newP(3,1)], [newP(2,2) newP(3,2)], 'Color',...
        Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
        'Parent', Haxes);
    rec(3) = line([newP(3,1) newP(4,1)], [newP(3,2) newP(4,2)], 'Color',...
        Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
        'Parent', Haxes);
    rec(4) = line([newP(4,1) newP(1,1)], [newP(4,2) newP(1,2)], 'Color',...
        Color, 'LineWidth', LineWidth, 'LineStyle', LineStyle,...
        'Parent', Haxes);                 
    
end

function res = check_Corners ( corners ) 
    res = all(size(corners) == [4 2]);
end

function res = check_T ( T ) 
    res = all(size(T) == [3 3]);
end

function res = check_Color ( Color ) 
    res = numel( Color ) == 3;
end