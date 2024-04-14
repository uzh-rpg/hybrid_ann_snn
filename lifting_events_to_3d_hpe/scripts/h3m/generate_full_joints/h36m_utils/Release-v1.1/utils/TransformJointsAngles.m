function newW = TransformJointsAngles( Wangle, WPos, Trans, Rot )
    newW = zeros(size(Wangle,1), 3);
    dir = [0 0 1] * Rot(1:3,1:3);
    for i=1:size(Wangle,1)
        R1 = vrrotvec2mat(vrrotvec(dir, (WPos(i,:)-Trans)/norm(WPos(i,:)-Trans)));
        dcm = angle2dcm(deg2rad(Wangle(i,1)), deg2rad(Wangle(i,2)), deg2rad(Wangle(i,3)), 'ZXY');
        dcm = dcm * Rot(1:3,1:3)';% * R1;
        [wx wy wz] = dcm2angle(dcm, 'ZXY');
        newW(i,:) = radtodeg([wx wy wz]);
    end
end