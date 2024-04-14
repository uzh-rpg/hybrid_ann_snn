% Copyright (C) 2010 Joao Carreira
%
% This code is part of the extended implementation of the paper:
% 
% J. Carreira, C. Sminchisescu, Constrained Parametric Min-Cuts for Automatic Object Segmentation, IEEE CVPR 2010
% 

function [bm bv] = my_BinMatrix(A,E,G,angle,bin)
% my_BINMATRIX Computes a Matrix (bm) with the same size of the image where
% (i,j) position contains the histogram value for the pixel at position (i,j)
% and another matrix (bv) where the position (i,j) contains the gradient
% value for the pixel at position (i,j)
%                
%IN:
%	A - Matrix containing the angle values
%	E - Edge Image
%   G - Matrix containing the gradient values
%	angle - 180 or 360%   
%   bin - Number of bins on the histogram 
%	angle - 180 or 360
%OUT:
%	bm - matrix with the histogram values
%   bv - matrix with the gradient values (only for the pixels belonging to
%   and edge)
% Originally from Anna Bosch, changed extensively by Joao Carreira

    X = size(E,2);
    Y = size(E,1);
    bm = zeros(Y,X);
    bv = zeros(Y,X);
    
    %bm2 = zeros(Y,X);
    %bv2 = zeros(Y,X);

    nAngle = angle/bin;
    
%     t = tic();
%     [posY,posX] = find(E);
%     for j=1:size(posY,1)
%         pos_x = posX(j,1);
%         pos_y = posY(j,1);
% 
%         b = ceil(A(pos_y,pos_x)/nAngle);
%         if G(pos_y,pos_x)>0
%             bm(pos_y,pos_x) = b;
%             bv(pos_y,pos_x) = G(pos_y,pos_x);
%         end
%     end
%     toc(t)
    
%    t = tic();
    b = ceil(A/nAngle);
    idx = E>0 & G>0;
    bm(idx) = b(idx);
    bv(idx) = G(idx);
%    toc(t);
%    ;
