function moveRectangle ( obj, corners, T, rec )   
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

    newP = obj.transformPoints ( corners, T );
    
    set(rec(1), 'XData', [newP(1,1) newP(2,1)], ...
                'YData', [newP(1,2) newP(2,2)]);    
            
    set(rec(2), 'XData', [newP(2,1) newP(3,1)], ...
                'YData', [newP(2,2) newP(3,2)]);
            
    set(rec(3), 'XData', [newP(3,1) newP(4,1)], ...
                'YData', [newP(3,2) newP(4,2)]);
            
    set(rec(4), 'XData', [newP(4,1) newP(1,1)], ...
                'YData', [newP(4,2) newP(1,2)]);   
            
    if ( numel(rec) == 6 )
        set(rec(5), 'XData', [newP(1,1) newP(3,1)], ...
                    'YData', [newP(1,2) newP(3,2)]);   
        set(rec(6), 'XData', [newP(2,1) newP(4,1)], ...
                    'YData', [newP(2,2) newP(4,2)]);   
    end
end