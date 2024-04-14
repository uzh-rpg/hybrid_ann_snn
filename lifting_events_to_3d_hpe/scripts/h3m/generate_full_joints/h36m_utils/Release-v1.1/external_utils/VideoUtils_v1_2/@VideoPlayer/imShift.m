%%%
%%% IMSHIFT: shift an image horizontally and/or vertically with wraparound
%%%   res = imshift( im, offset )
%%%     im: grayscale or color image
%%%     offset: [dY dX]
%%%   Hany Farid; Image Science Group; Dartmouth College
%%%   10.13.06
%%%

function  res = imShift( obj, im, offset )

   dims = size(im);
   
   offset = -offset;
   
   if offset(1) > 0
       im(:, 1:abs(offset(1)), :) = 0;
   end
   
   if offset(1) < 0
       im(:, (end - abs(offset(1))):end, :) = 0;
   end
   
   if offset(2) > 0
       im(1:abs(offset(2)), :, :) = 0;
   end
   
   if offset(2) < 0
       im((end - abs(offset(2))):end, :, :) = 0;
   end
   
   offset = mod(offset, [dims(2) dims(1)]);
   res = zeros( dims );

   res(:,:,:) = [ im(offset(2)+1:dims(1), offset(1)+1:dims(2), :),  ...
                  im(offset(2)+1:dims(1), 1:offset(1), :); ...
                  im(1:offset(2), offset(1)+1:dims(2), :), ...
                  im(1:offset(2), 1:offset(1), :) ];

end