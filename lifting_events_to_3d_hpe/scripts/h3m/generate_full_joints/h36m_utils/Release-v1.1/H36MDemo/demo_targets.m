%% this example is set up to exemplify the use of our target data in different
% formats as well as the way to transform from one format to another
%
% pose data is available in angles or positions parametrizations, 2d or 3d, original or 
% monocular parametrization (cameras superposed) and original version or mirror 
% symmetric with respect to the camera view
%
% mirror symmetric versions of both images and masks can also be easily obtained

% Setup
addpaths;
clear;
close all;

Features{1} = H36MPose3DAnglesFeature(); 
Features{2} = H36MPose3DPositionsFeature(); 
Features{3} = H36MPose2DPositionsFeature();
Features{4} = H36MPose3DPositionsFeature('Monocular',true);
Features{5} = H36MPose3DPositionsFeature('Symmetric',true);
Features{6} = H36MPose2DPositionsFeature('Symmetric',true);
Features{7} = H36MPose3DPositionsFeature('Monocular',true,'Symmetric',true);
Features{8} = H36MPose3DAnglesFeature('Monocular',true); 

% select the data
fno = 800;
Sequence = H36MSequence(1,13,2,1,fno);
F = H36MComputeFeatures(Sequence, Features);

vidfeat = H36MRGBVideoFeature();
da = vidfeat.serializer(Sequence);
im = da.getFrame(fno);
flim = flipdim(im,2);

Subject = Sequence.getSubject();
Camera = Sequence.getCamera();
Camera0 = H36MCamera(H36MDataBase.instance(), 0,1);

posSkel   = Subject.getPosSkel();
angSkel   = Subject.getAnglesSkel();
pos2dSkel = Subject.get2DPosSkel();
figure(10); showPose(F{4},posSkel); axis ij

%% original data
% 3d angles
figure(1); imshow(im); show2DPose(Camera.project(Features{1}.toPositions(F{1},angSkel)),pos2dSkel); title('3d angles ');

% 3d positions
figure(2); imshow(im); show2DPose(Camera.project(F{2}),pos2dSkel); title('3d positions ');


% 2d positions
figure(3); imshow(im); show2DPose(F{3},pos2dSkel); title('2d positions');


%% monocular data
% 3d angles
figure(8); imshow(im); show2DPose(Camera0.project(Features{8}.toPositions(F{8},angSkel)),pos2dSkel); title('3d angles monocular');

% 3d positions
figure(4); imshow(im); show2DPose(Camera0.project(F{4}),pos2dSkel); title('monocular positions 3d');

%% mirror symmetric
% 3d positions
figure(5); imshow(flim); show2DPose(Camera.project(F{5}),pos2dSkel); title('mirror symmetric 3d positions');

% 2d positions
figure(6); imshow(flim); show2DPose(F{6},pos2dSkel); title('mirror symmetric 2d positions');

%% monocular + mirror symmetric
% 3d positions
figure(7); imshow(flim); show2DPose(Camera0.project(F{7}),pos2dSkel); title('monocular mirror symmetric 3d positions');
