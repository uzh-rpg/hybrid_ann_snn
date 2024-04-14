function p = phog_backmasked(Img,bin,angle,L,rois, masks, pb_file, just_bbox)
% Adapted from Anna Bosch's code, by Joao Carreira, July 9 2009
% adapted for pb_file

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

    DefaultVal('*just_bbox', 'false');
    
    if size(Img,3) == 3
        G = rgb2gray(Img);
    else
        G = Img;
    end    
    
    [GradientX,GradientY] = gradient(double(G));    
    index = GradientX == 0;
    GradientX(index) = 1e-5;
    
    YX = GradientY./GradientX;
    if angle == 180, A = ((atan(YX)+(pi/2))*180)/pi; end
    if angle == 360, A = ((atan2(GradientY,GradientX)+pi)*180)/pi; end

    % MODIF catalin 20.01
    if(exist('pb_file', 'var') && ~isempty(pb_file)) && ischar(pb_file)
        load(pb_file);
        Gr = gPb_thin;
        E = (gPb_thin>0.01);
    else
        if isempty(pb_file)
          E = edge(G,'canny', [0.01 0.2]);
        else
          E = pb_file;
        end
        %E = edge(G,'canny', [0.01 0.05]);
        Gr = sqrt((GradientX.*GradientX)+(GradientY.*GradientY));
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%
    % the roi might be outside the image, then 
    % we need to grow all the stuff, adding a frame of zeros:
    %   - A, E, Gr    
    sz_Img = size(Img);
    up = abs(min(0,min(rois(1,:))));
    down = max(0,max(rois(2,:)) - sz_Img(1));
    left = abs(min(0,min(rois(3,:))));
    right = max(0,max(rois(4,:)) - sz_Img(2));
    A = grow_it(A, up, down, left, right); 
    E = grow_it(E, up, down, left, right); 
    Gr = grow_it(Gr, up, down, left, right); 
    
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
        % grow mask a little bit
        %t = tic();
        if(~just_bbox)
            mask = masks(:,:,i);        
            sel = strel('diamond',3);
            dilated_mask = imdilate(mask, sel);

            % now grow it as the others
            dilated_mask = grow_it(dilated_mask, up, down, left, right); 

            E_mask = E.*dilated_mask;
            Gr_mask = Gr.*dilated_mask;
        else
            E_mask = E;
            Gr_mask = Gr;
        end
        
        A_roi = A(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        E_roi = E_mask(rois(1,i):rois(2,i),rois(3,i):rois(4,i));
        G_roi = Gr_mask(rois(1,i):rois(2,i),rois(3,i):rois(4,i));

        [bh_roi bv_roi] = my_BinMatrix(A_roi,E_roi,G_roi,angle,bin);
        
        %p(:,i) = my_phog_descriptor(bh_roi,bv_roi,L,bin);
        p(:,i) = my_phog_desc_mex(bh_roi,bv_roi,L,bin); % andreas mex file
        
        %toc(t)
        %draw_descriptor_last_level(p(end-bin*64 +1:end,i)', bv_roi, bin, 8);
        %subplot(1,3,1), imshow(bh_roi);
        %subplot(1,3,2), imshow(E.*mask);
        %subplot(1,3,3), imshow(mask);
    end
end

function newI = grow_it(I, up, down, left, right)
    new_size = [(size(I,1) + up + down) size(I,2) + left + right];
    
    newI = zeros(new_size);
    newI(up+1:end-down, left+1:end-right) = I;
end
