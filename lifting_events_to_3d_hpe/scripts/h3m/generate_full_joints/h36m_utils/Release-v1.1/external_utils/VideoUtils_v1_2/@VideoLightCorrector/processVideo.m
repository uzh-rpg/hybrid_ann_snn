function processVideo( obj )
%VIDEOLIGTHCORRECTOR Class for reading videos and correction subit
%light changes, using a polynomial aproach.
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

    vp = VideoPlayer(obj.VideoName, obj.VPParams{:});

    numPixels = vp.Width * vp.Height;
    
    obj.Original   = zeros([vp.NumFrames 1]);
    obj.Correction = zeros([vp.NumFrames 1]);
    obj.Modified   = zeros([vp.NumFrames 1]);
    
    for i = 1:vp.NumFrames
        plot(vp);
        
        grayIma = rgb2gray(vp.Frame);
        obj.Original(i) = sum(grayIma(:)) / numPixels;
        
        vp.nextFrame;
        
        drawnow;
    end

    x = 1:vp.NumFrames;
    factor = polyfit(x', obj.Original, obj.Grade);
    f = polyval(factor, x');
    
    eval(['close ' num2str(vp.Hfig)])
    clear vp;
    vp = VideoPlayer(obj.VideoName, obj.VPParams{:});
    
    hf = figure; hima = imshow(vp.Frame); title('Left Side Original Frame, Righ Side New Frame');
    
    for i = 1:vp.NumFrames
        plot(vp);

        obj.Correction(i) = f(i) / obj.Original(i);
        frame = vp.Frame * obj.Correction(i);

        gray = rgb2gray(frame);
        obj.Modified(i) = sum(gray(:)) / numPixels;

        frame(frame > 1 ) = 1;
        frame(frame < 0 ) = 0;
   
        frame(:, 1:vp.Width / 2, :) = vp.Frame(:, 1:vp.Width / 2, :);
   
        set(hima, 'cdata',  frame);
        drawnow
   
        vp.nextFrame;
        
        drawnow;
    end
    
    nameFile   = [obj.VideoName(1:end - 4) '_LC' num2str(obj.Grade) '.mat'];
    correction = obj.Correction;
    original   = obj.Original;
    modified   = obj.Modified;
    obj.Function = f;
    
    figure, plot(original, 'b');
    hold on;
    plot(f, 'r');
    plot(modified, 'g');
    hold off;
    
    eval(['close ' num2str(vp.Hfig)]);
    eval(['close ' num2str(hf)]);
    clear vp;
    
    
    eval(['save ' nameFile ' correction original modified f;']);
end