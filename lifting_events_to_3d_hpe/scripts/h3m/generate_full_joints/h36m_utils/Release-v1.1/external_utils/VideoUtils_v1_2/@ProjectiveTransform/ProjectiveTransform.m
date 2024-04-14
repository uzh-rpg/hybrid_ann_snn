classdef  ...
        ( ...
          Hidden = false, ...          
          InferiorClasses = {}, ...    
          ConstructOnLoad = false, ... 
          Sealed = false ...           
        ) ProjectiveTransform < handle
%   ProjectiveTransform: Class for applying projective transformations to
%   points and images.
%
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 15 $
%   $Date: 2012-04-25 19:09:46 +0200 (Wed, 25 Apr 2012) $
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

    properties (GetAccess='public', SetAccess='public')       
        
    end
        
    properties (GetAccess='public', SetAccess='protected')       
        Points2TType = 'Inhomogeneous';
    end
    
    properties (Hidden, Access = private)

    end
    
    methods
        
        % Constructor
        function obj = ProjectiveTransform ( varargin )

        end
    
    end
    
    methods (Access = public)
        newPoints = transformPoints ( obj, points, T );
        
        bbox = findBoundingBox ( obj, corners );
        
        [T Tinv]  = points2T ( obj, points, base_points ); 
        [T Tinv]  = points2T_Matlab ( obj, points, base_points )
        [T Tinv]  = points2T_DLT ( obj, points, base_points ); 
        [T Tinv]  = points2T_Inhomogeneous ( obj, points, base_points );
        
        [T Tinv]  = points2AffineT_Inhomogeneous ( obj, points, base_points );
        
        [T Tinv]  = points2T_Weighted ( obj, points, base_points, weight ); 
        [T Tinv]  = points2T_Weighted_Inhomogeneous ( obj, points, base_points, weight );
        
        varargout = transformImage(obj, Image, T, varargin );
        T = createTMatrix ( obj, trans );
        [Tf Tfinv] = getTf ( obj, image );
        [IFcorners IGcorners center] = getImageCorners ( obj, image );
        
        rec = drawRectangle ( obj, Corners, T, varargin );
        moveRectangle ( obj, corners, T, hrec );
        [Tm corners sizeM] = getTm ( obj, Pgp );
    end
    
    methods (Hidden, Access = private)
        [newpts, T] = normalise2DPts(obj, pts);
    end
    
    methods (Hidden, Access = public)
        check_param_Points2TType ( obj, value );
%         check_param_<name> ( obj, value );
    end
    
    methods (Access = private)       
    end
end