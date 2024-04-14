function [T Tinv]  = points2AffineT_Inhomogeneous ( obj, points, base_points )
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

    Npts = length(points);
    
    x1 = base_points;
    x2 = points;
    
    A = zeros([Npts * 2, 6]);
    X_ = zeros([Npts * 2, 1]);
    
    for i = 1:Npts
        x = x1(i, 1);
        y = x1(i, 2);
        
        u = x2(i, 1);
        v = x2(i, 2);
        
        u_ = [x y 1 0 0 0];
        v_ = [0 0 0 x y 1];
        
        n = i * 2;
        A(n - 1, :) = u_;
        A(n, :)     = v_;
        
        X_(n - 1) = u;
        X_(n)     = v;
    end
    
    h_ = (((A'*A)^-1)*A') * X_;
    
    Tinv = reshape([h_; 0; 0; 1], 3, 3);
    T = Tinv^-1;
    T = T / T(3, 3);
end