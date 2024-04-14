function T = createTMatrix ( obj, trans )
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

    transX = trans(1);
    transY = trans(2);

    rotX = trans(3);
    rotY = trans(4);
    rotZ = trans(5);
    
    scale = trans(6);

    radRotX = rotX * (pi / 180.0);
    radRotY = rotY * (pi / 180.0);
    radRotZ = rotZ * (pi / 180.0);
    
    transl = [     1      0 transX; ...
                   0      1 transY; ...
                   0      0      1];   

    matRotX = [1, 0, 0; ...
               0, cos(radRotX), -sin(radRotX); ...
               0, sin(radRotX),  cos(radRotX)];

    matRotY = [ cos(radRotY), 0, sin(radRotY); ...
                           0, 1,            0; ...
               -sin(radRotY), 0, cos(radRotY)];

    matRotZ = [cos(radRotZ), -sin(radRotZ), 0; ...
               sin(radRotZ), cos(radRotZ), 0; ...
               0, 0, 1];

    matScale = [scale 0 0;
                0 scale 0;
                0 0 1];

    T = (transl * matRotX * matRotY * matRotZ * matScale)';
    T = T / T(3, 3);
end