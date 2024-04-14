function p = phog_masked(masks,bin,angle,L,rois, pb_file)
% Adaptation of Anna Bosch's code, by Joao Carreira, July 9 2009, July 5 2010
%

%IN:
%	I - Images of size MxN (Color or Gray)
%	bin - Number of bins on the histogram 
%	angle - 180 or 360
%   L - number of pyramid levels
%   rois - Regions Of Interest (ytop,ybottom,xleft,xright)
%   masks - the masks
%
%OUT:
%	p - pyramid histogram of oriented gradients

    %%%%%%%%%%%%%%%%%%%%%%%%
    % the roi might be outside the image, then 
    % we need to grow all the stuff, adding a frame of zeros:
    %   - A, E, Gr    
    sz_Img = size(masks); 
    up = abs(min(0,min(rois(1,:))));
    down = max(0,max(rois(2,:)) - sz_Img(1));
    left = abs(min(0,min(rois(3,:))));
    right = max(0,max(rois(4,:)) - sz_Img(2));
    
    masks = grow_it(masks, up, down, left, right);
    rois(1,:) = rois(1,:) + up + 1;
    rois(2,:) = rois(2,:) + up;
    rois(3,:) = rois(3,:) + left + 1;
    rois(4,:) = rois(4,:) + left;
    %%%%%%%%%%%%%%%%%%%%%%%%%

        
    if(L == 2)
        dim = bin + 4*bin+16*bin;
    elseif(L == 3)
        dim = bin + 4*bin + 16*bin + 64*bin;
    end
    p = zeros(dim,size(masks,3)); % 420 is size of descriptor for a 3 level pyramid ( L == 2)
    for i=1:size(masks,3)
        %t = tic();
        G = masks(:,:,i);
        
        G = G(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        [GradientX,GradientY] = gradient(double(G));
        
        index = GradientX == 0;
        GradientX(index) = 1e-5;
        
        YX = GradientY./GradientX;
        if angle == 180, A = ((atan(YX)+(pi/2))*180)/pi; end
        if angle == 360, A = ((atan2(GradientY,GradientX)+pi)*180)/pi; end
        
         
        Gr = sqrt((GradientX.*GradientX)+(GradientY.*GradientY));

        ids_in_E = bwboundaries(G);
        ids_in_E = cell2mat(ids_in_E);
        E = zeros(size(G));
        
        if(~isempty(ids_in_E))
            E(sub2ind(size(E), ids_in_E(:,1), ids_in_E(:,2))) = 1;
        end
        
        %A_roi = A(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        %E_roi = E(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        %G_roi = Gr(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        A_roi = A;
        E_roi = E;
        G_roi = Gr;
        
        [bh_roi bv_roi] = my_BinMatrix(A_roi,E_roi,G_roi,angle,bin);
                
        p(:,i) = my_phog_desc_mex(bh_roi,bv_roi,L,bin); % andreas mex file
        %p(:,i) = my_phog_descriptor(bh_roi,bv_roi,L,bin);        
        
        %draw_descriptor_last_level(p(end-bin*64 +1:end,i)', bv_roi, bin, 8);
        %subplot(1,3,1), imshow(bh_roi);
        %subplot(1,3,2), imshow(E.*mask);
        %subplot(1,3,3), imshow(mask);
        %toc(t)
    end
    p = single(p);
end

function newI = grow_it(I, up, down, left, right)
    new_size = [(size(I,1) + up + down) size(I,2) + left + right size(I,3)];
    
    %newI = zeros(new_size);
    newI = false(new_size);
    newI(up+1:end-down, left+1:end-right,:) = I;
end