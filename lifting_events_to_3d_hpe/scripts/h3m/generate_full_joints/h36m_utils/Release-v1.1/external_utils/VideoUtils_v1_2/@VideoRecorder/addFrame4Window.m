function resFi = addFrame4Window(obj, frame1, frame2, frame3, frame4)    
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

    [h w c] = size(frame1);

    if ( c ~= 3 )
        frameAux1 = obj.convert2ThreeChannels(frame1);
    else
        frameAux1 = frame1;
    end

    for i = 2:4
        eval(['cAux = size(frame' num2str(i) ', 3);'])

        if ( cAux ~= 3 )
            eval(['frameAux' num2str(i) ' = obj.convert2ThreeChannels(frame' num2str(i) ');'])
        else
            eval(['frameAux' num2str(i) ' = frame' num2str(i) ';']);
        end
    end

    resFi = zeros([h * 2 w * 2 c]);

    h1 = 1:h;
    w1 = 1:w;

    h2 = (h+1):(h*2);
    w2 = (w+1):(w*2);

    resFi(h1, w1, :) = frameAux1;
    resFi(h1, w2, :) = frameAux2;
    resFi(h2, w1, :) = frameAux3;
    resFi(h2, w2, :) = frameAux4;

    resFi ((h - 1):(h + 1), :, 1) = 1;
    resFi (:, (w - 1):(w + 1), 1) = 1;

    resFi ((h - 1):(h + 1), :, 2) = 0;
    resFi (:, (w - 1):(w + 1), 2) = 0;

    resFi ((h - 1):(h + 1), :, 3) = 0;
    resFi (:, (w - 1):(w + 1), 3) = 0;

    obj.addFrame( resFi );
end