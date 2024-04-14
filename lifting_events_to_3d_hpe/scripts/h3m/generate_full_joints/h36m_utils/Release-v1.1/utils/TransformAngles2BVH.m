function BVHFrameByRot = TransformAngles2BVH(Pred, frame, orig)
  CSVDoF =     [      3                   1               3               1               3               1               3                 1             3                   3               3            3             3                     3                       1            3               3                3                       1                  3 ];
  BVH_2_CSV = [0 1:4 0 5:8 0 9:12 0 13:16 -1 0 -1 0 17:20 -1 0 -1 0];
  N = length(frame);
  BVHFrameByRot = zeros(N, 78);
  if ~exist('orig','var')
    BVHFrameByRot(:,1:6) = [repmat([0 91.0641 0],N,1) Pred(frame,1:3)];
  else
    BVHFrameByRot(:,1:6) = [orig(frame,:) Pred(frame,1:3)];
  end
  ind = 4;
  frameind = 7;
  for j = 2:length(BVH_2_CSV)
    csvindex = BVH_2_CSV(j);
    if(csvindex==0)
       continue
    end
    if(csvindex==-1)
        BVHFrameByRot(frame,frameind:frameind+2) = zeros(N,3); %%%%% 0DOF
        frameind = frameind+3;
        continue
    end
    if(j>=2 && j<=5) % EXCHANGE LEFT WITH RIGHT LEG
        if CSVDoF(csvindex) == 3
          BVHFrameByRot(:,frameind:frameind+2) = Pred(frame,ind+8:ind+10); %%%%% 3DOF
          frameind = frameind+3;
          ind = ind+3;
        else 
          BVHFrameByRot(:,frameind:frameind+2) = [zeros(N,1) Pred(frame,ind+8) zeros(N,1)]; %%%%% 1DOF
          frameind = frameind+3;
          ind = ind+1;
        end
    elseif(j>=7 && j<=10) % EXCHANGE RIGHT WITH LEFT LEG
        if CSVDoF(csvindex) == 3
          BVHFrameByRot(:,frameind:frameind+2) = Pred(frame,ind-8:ind-6); %%%%% 3DOF
          frameind = frameind+3;
          ind = ind+3;
        else 
          BVHFrameByRot(:,frameind:frameind+2) = [zeros(N,1) Pred(frame,ind-8) zeros(N,1)]; %%%%% 1DOF
          frameind = frameind+3;
          ind = ind+1;
        end
    else
        if CSVDoF(csvindex) == 3
          BVHFrameByRot(:,frameind:frameind+2) = Pred(frame,ind:ind+2); %%%%% 3DOF
          frameind = frameind+3;
          ind = ind+3;
        else 
          BVHFrameByRot(:,frameind:frameind+2) = [zeros(N,1) Pred(frame,ind) zeros(N,1)]; %%%%% 1DOF
          frameind = frameind+3;
          ind = ind+1;
        end
    end
  end
  BVHFrameByRot(:,[14 26]) = BVHFrameByRot(:,[14 26])+15;
  BVHFrameByRot(:,[17 29]) = BVHFrameByRot(:,[17 29])-15;
  BVHFrameByRot(:,43) = BVHFrameByRot(:,43)-90;
  BVHFrameByRot(:,61) = BVHFrameByRot(:,61)+90;
end
