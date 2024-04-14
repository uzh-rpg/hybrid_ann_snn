% HOMOGRAPHY2D - computes 2D homography
%
% Usage:   H = homography2d(x1, x2)
%          H = homography2d(x)
%
% Arguments:
%          x1  - 3xN set of homogeneous points
%          x2  - 3xN set of homogeneous points such that x1<->x2
%         
%           x  - If a single argument is supplied it is assumed that it
%                is in the form x = [x1; x2]
% Returns:
%          H - the 3x3 homography such that x2 = H*x1
%
% This code follows the normalised direct linear transformation 
% algorithm given by Hartley and Zisserman "Multiple View Geometry in
% Computer Vision" p92.
%

% Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% May 2003  - Original version.
% Feb 2004  - Single argument allowed for to enable use with RANSAC.
% Feb 2005  - SVD changed to 'Economy' decomposition (thanks to Paul
% O'Leary)

function [T Tinv]  = points2T_DLT ( obj, points, base_points )
    Npts = length(points);
    
    newP  = [points ones([Npts, 1])];
    newBP = [base_points ones([Npts, 1])];
    
    % Attempt to normalise each set of points so that the origin 
    % is at centroid and mean distance from origin is sqrt(2).
    [x1, T1] = obj.normalise2DPts(newBP');
    [x2, T2] = obj.normalise2DPts(newP');
    
    % Note that it may have not been possible to normalise
    % the points if one was at infinity so the following does not
    % assume that scale parameter w = 1.
    
    
%     A = zeros(3*Npts,9);
%     
%     O = [0 0 0];
%     for n = 1:Npts
%         X = x1(:,n)';
%         x = x2(1,n); y = x2(2,n); w = x2(3,n);
%         A(3*n-2,:) = [  O  -w*X  y*X];
%         A(3*n-1,:) = [ w*X   O  -x*X];
%         A(3*n  ,:) = [-y*X  x*X   O ];
%     end
    
    A = zeros(2*Npts,9);
    
    O = [0 0 0];
    for n = 1:Npts
        X = x1(:,n)';
        x = x2(1,n); y = x2(2,n); w = x2(3,n);
        A(2*n-1,:) = [  O  -w*X  y*X];
        A(2*n,:)   = [ w*X   O  -x*X];
    end
    
    [~,~,V] = svd(A, 0); % 'Economy' decomposition for speed
    
    % Extract homography
    H = reshape(V(:,9),3,3)';
    
    % Denormalise
    Tinv = ((T2^-1)*H*T1)';   
    Tinv = Tinv / Tinv(3, 3);
    T = Tinv^-1;
    T = T / T(3, 3);
end