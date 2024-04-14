function final = stitchMask ( obj, I1, I2, mask )
% Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
% All rights reserved.
%
%   $Revision: 16 $
%   $Date: 2012-04-28 13:45:12 +0200 (Sat, 28 Apr 2012) $
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

    [h w c] = size(I1);
    
    mask = obj.createMaskPyramidMask( w, h, mask );

    data1r =  obj.createGaussLaplaPyramid(I2(:, :, 1));
    
    data2r =  obj.createGaussLaplaPyramid(I1(:, :, 1));

    resr = obj.stitch(data1r, data2r, mask);

    data1g =  obj.createGaussLaplaPyramid(I2(:, :, 2));
    data2g =  obj.createGaussLaplaPyramid(I1(:, :, 2));

    resg = obj.stitch(data1g, data2g, mask);

    data1b =  obj.createGaussLaplaPyramid(I2(:, :, 3));
    data2b =  obj.createGaussLaplaPyramid(I1(:, :, 3));

    resb = obj.stitch(data1b, data2b, mask);

    final = zeros(size(I1));
    final(:, :, 1) = resr;
    final(:, :, 2) = resg;
    final(:, :, 3) = resb;
    
    final(final > 1) = 1;
    final(final < 0) = 0;

    if ( obj.ShowIntermediate )
        for i = 2:obj.NumLevels
            ima1 = zeros([data1r{i}.height data1r{i}.width 3]);

            ima1(:, :, 1) = data1r{i}.gaussian;
            ima1(:, :, 2) = data1g{i}.gaussian;
            ima1(:, :, 3) = data1b{i}.gaussian;

            ima2 = zeros([data1r{i}.height data1r{i}.width 3]);

            ima2(:, :, 1) = data2r{i}.gaussian;
            ima2(:, :, 2) = data2g{i}.gaussian;
            ima2(:, :, 3) = data2b{i}.gaussian;

            imwrite(ima1, [ 'Marc_Gaussian_' num2str(i) '.jpg']);
            imwrite(ima2, ['Oriol_Gaussian_' num2str(i) '.jpg']);

            imwrite(mask{i}.mask, ['mask_Gaussian_' num2str(i) '.jpg']);
        end

        for i = 1:obj.NumLevels-1
            ima1 = zeros([data1r{i}.height data1r{i}.width 3]);

            ima1(:, :, 1) = data1r{i}.laplacian;
            ima1(:, :, 2) = data1g{i}.laplacian;
            ima1(:, :, 3) = data1b{i}.laplacian;

            ima2 = zeros([data1r{i}.height data1r{i}.width 3]);

            ima2(:, :, 1) = data2r{i}.laplacian;
            ima2(:, :, 2) = data2g{i}.laplacian;
            ima2(:, :, 3) = data2b{i}.laplacian;

            tire_imadjust1 = ima1;
            tire_imadjust2 = ima2;

            tire_imadjust1(:, :, 1) = imadjust(ima1(:, :, 1));
            tire_imadjust1(:, :, 2) = imadjust(ima1(:, :, 2));
            tire_imadjust1(:, :, 3) = imadjust(ima1(:, :, 3));
            tire_imadjust2(:, :, 1) = imadjust(ima2(:, :, 1));
            tire_imadjust2(:, :, 2) = imadjust(ima2(:, :, 2));
            tire_imadjust2(:, :, 3) = imadjust(ima2(:, :, 3));

            imwrite(tire_imadjust1, [ 'Marc_Laplacian_imadjust_' num2str(i) '.jpg']);
            imwrite(tire_imadjust2, ['Oriol_Laplacian_imadjust_' num2str(i) '.jpg']);

            ire_histeq1 = ima1;
            ire_histeq2 = ima2;

            ire_histeq1(:, :, 1) = histeq(ima1(:, :, 1));
            ire_histeq1(:, :, 2) = histeq(ima1(:, :, 2));
            ire_histeq1(:, :, 3) = histeq(ima1(:, :, 3));
            ire_histeq2(:, :, 1) = histeq(ima2(:, :, 1));
            ire_histeq2(:, :, 2) = histeq(ima2(:, :, 2));
            ire_histeq2(:, :, 3) = histeq(ima2(:, :, 3));

            imwrite(ire_histeq1, [ 'Marc_Laplacian_histeq_' num2str(i) '.jpg']);
            imwrite(ire_histeq2, ['Oriol_Laplacian_histeq_' num2str(i) '.jpg']);

            tire_adapthisteq1 = ima1;
            tire_adapthisteq2 = ima2;

            tire_adapthisteq1(:, :, 1) = adapthisteq(ima1(:, :, 1));
            tire_adapthisteq1(:, :, 2) = adapthisteq(ima1(:, :, 2));
            tire_adapthisteq1(:, :, 3) = adapthisteq(ima1(:, :, 3));
            tire_adapthisteq2(:, :, 1) = adapthisteq(ima2(:, :, 1));
            tire_adapthisteq2(:, :, 2) = adapthisteq(ima2(:, :, 2));
            tire_adapthisteq2(:, :, 3) = adapthisteq(ima2(:, :, 3));

            imwrite(tire_adapthisteq1, [ 'Marc_Laplacian_adapthisteq_' num2str(i) '.jpg']);
            imwrite(tire_adapthisteq2, ['Oriol_Laplacian_adapthisteq_' num2str(i) '.jpg']);
        end
    end
end