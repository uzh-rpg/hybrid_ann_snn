function final = stitchLineal ( obj, I1, I2 )
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

    [h w c] = size(I1);
    
    mask = obj.createLinealPyramidMask( w, h );

    data1r =  obj.createGaussLaplaPyramid(I2(:, :, 1));
    data2r =  obj.createGaussLaplaPyramid(I1(:, :, 1));

    resr = obj.stitch(data1r, data2r, mask);

    data1g =  obj.createGaussLaplaPyramid(I2(:, :, 2));
    data2g =  obj.createGaussLaplaPyramid(I1(:, :, 2));

    resg = obj.stitch(data1g, data2g, mask);

    data1b =  obj.createGaussLaplaPyramid(I2(:, :, 3));
    data2b =  obj.createGaussLaplaPyramid(I1(:, :, 3));

    resb = obj.stitch(data1b, data2b, mask);

    final = zeros(size(I1));
    final(:, :, 1) = resr;
    final(:, :, 2) = resg;
    final(:, :, 3) = resb;
    
    final(final > 1) = 1;
    final(final < 0) = 0;
end