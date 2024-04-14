function [BW bbox] = preproc_mask(mask, scc)
% PREPROC_MASK(MASK, SCC) 
% preprocessing with or without scc, computes bounding box for efficiency
  if scc
		% largest strongly connected component
    CC = bwconncomp(mask,8);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    BW = zeros(size(mask));
    BW(CC.PixelIdxList{idx}) = mask(CC.PixelIdxList{idx});
    
    p = regionprops(BW, 'BoundingBox');
    bbox = p.BoundingBox;
  else
    p = regionprops(mask, 'BoundingBox');
    for i = 1: length(p)
      boxes(i,:) = [p(i).BoundingBox(1:2) p(i).BoundingBox(1:2)+p(i).BoundingBox(3:4)];
    end
    mins = min(boxes,[],1); maxs = max(boxes,[],1);
    bbox= [mins(1) mins(2) maxs(3)-mins(1) maxs(4)-mins(2)];
    
    BW = mask;
  end
  
end