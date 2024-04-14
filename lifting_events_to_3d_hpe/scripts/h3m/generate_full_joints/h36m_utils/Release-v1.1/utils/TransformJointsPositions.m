function Joints = TransformJointsPositions( Joints, WPos, Trans, Rot, direct )
% Transforms joints positions relative to each camera in the same coordinate system
% Joints : joint positions in the global coordinate system
% WPos : global root
% Trans : camera translation
% Rot : camera rotation (3x3 matrix format)
if ~exist('direct','var') || direct
  dir = [0 0 1] * Rot(1:3,1:3);
  for i=1:size(Joints,1)
    R1 = vrrotvec2mat(vrrotvec(dir, (WPos(i,:)-Trans)/norm(WPos(i,:)-Trans)))';
     R1 = eye(3);
    for j=1:3:size(Joints,2)
      Joints(i,j:j+2)  = (Joints(i,j:j+2)-Trans) * Rot' * R1;
    end
  end
else
  dir = [0 0 1] * Rot(1:3,1:3);
  for i=1:size(Joints,1)
    R1t = vrrotvec2mat(vrrotvec(dir, (WPos(i,:)-Trans)/norm(WPos(i,:)-Trans)));
     R1t = eye(3);
    for j=1:3:size(Joints,2)
      Joints(i,j:j+2) = Joints(i,j:j+2) * R1t * Rot + Trans;
    end
  end
end
