function example ( obj )
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

    sizeIm = [1280 960];

    % Load two images
    I1 = double(imread('Marc.jpg')) / 255;
    I1 = imresize ( I1, sizeIm);
    I1 ( I1 > 1 ) = 1;
    I1 ( I1 < 0 ) = 0;

    I2 = double(imread('Kike.jpg')) / 255;
    I2 = imresize ( I2, sizeIm);
    I2 ( I2 > 1 ) = 1;
    I2 ( I2 < 0 ) = 0;
    
    % Loadding binnary mask
    Imask = double(imread('Face.png')) / 255;
    mask = Imask(:, :, 1);
    mask = imresize ( mask, sizeIm);
    mask ( mask > 1 ) = 1;
    mask ( mask < 0 ) = 0;

    % Initialie Multi-band Blending
    mb = MultibandBlending('NumLevels', 7, 'MaskBlur', 6, 'ItPSF', 2);

    % Stitching examples.
    res = mb.stitchLineal( I1, I2 );
    figure, hima = imshow(res);
    pause;
    
    res = mb.stitchMaskDirect(I1, I2, mask);
    set(hima, 'cdata', res);
    pause;
    
    res = mb.stitchMask(I1, I2, mask);
    set(hima, 'cdata', res);
    pause;
    
    res = mb.stitchMaskDirect(I2, I1, mask);
    set(hima, 'cdata', res);
    pause;
    
    res = mb.stitchMask(I2, I1, mask);
    set(hima, 'cdata', res);
    pause;
    
    clear mb;
end