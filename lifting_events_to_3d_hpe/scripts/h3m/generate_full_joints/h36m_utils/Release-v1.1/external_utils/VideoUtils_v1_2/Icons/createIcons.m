% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 9 $
%   $Date: 2012-04-20 11:31:29 +0200 (Fri, 20 Apr 2012) $
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

clear

matIconName = 'VideoPlayerIcons';

files = dir;

count = 1;

for i = 1:numel(files)
    clear imact;
    
    indx = strfind(files(i).name, '.png');
    if (indx)
        imact = double(imread(files(i).name)) / 255.0;
        
        [h w c] = size(imact);
        
        tR = imact(:, :, 1) == 1.0;
        tG = imact(:, :, 2) == 0.0;
        tB = imact(:, :, 3) == 1.0;
        
        t = tR .* tG .* tB;
        
        r = imact(:, :, 1);
        g = imact(:, :, 2);
        b = imact(:, :, 3);
        
        r(t == 1) = NaN;
        g(t == 1) = NaN;
        b(t == 1) = NaN;
        
        res(:, :, 1) = r;
        res(:, :, 2) = g;
        res(:, :, 3) = b;

        eval([matIconName '.' files(i).name(1:(indx - 1)) ' = res;']);
    end
end

eval(['save ' matIconName]);
movefile([matIconName '.mat'], ['../' matIconName '.mat']);
