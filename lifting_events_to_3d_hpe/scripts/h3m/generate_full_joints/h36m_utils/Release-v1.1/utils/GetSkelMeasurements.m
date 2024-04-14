function skel = GetSkelMeasurements(skel, Positions, forMotionBuilder)
    links = [1 2; 2 3; 3 4; 4 5;... % left leg
         6 7; 7 8; 8 9; 9 10;... % right leg
         11 1; 11 6;... % hips (L/R)
         11 12; 12 13;... % spine
         13 14; 14 15;... % head
         13 17; 17 18; 18 19;... % left arm
         19 21; 19 22;... % left hand
         13 24; 24 25; 25 26;... % right arm
         26 28; 26 29]; % right hand
    for j=1:size(links,1)
        if(forMotionBuilder)
            if(j==1 || j==2) %Left leg
                skel.tree(1,j+7).offset = [0, -norm(skel.tree(1,j+7).offset), 0];
            elseif(j==3)
                skel.tree(1,j+7).offset = [0, 0, norm(skel.tree(1,j+7).offset)];
            elseif(j==4)
                skel.tree(1,j+7).offset = [0, 0, norm(skel.tree(1,j+7).offset)];
            elseif(j==5 || j==6) %Right leg
                skel.tree(1,j-2).offset = [0, -norm(skel.tree(1,j-2).offset), 0];
            elseif(j==7) 
                skel.tree(1,j-2).offset = [0, 0, norm(skel.tree(1,j-2).offset)];
            elseif(j==8)
                skel.tree(1,j-2).offset = [0, 0, norm(skel.tree(1,j-2).offset)];
            elseif(j==9) % LHips
                skel.tree(1,7).offset = [norm(skel.tree(1,7).offset), 0, 0];
            elseif(j==10) % RHips
                skel.tree(1,2).offset = [-norm(skel.tree(1,2).offset), 0, 0];
            elseif(j==11) % Spine
                skel.tree(1,13).offset = [0, norm(skel.tree(1,13).offset), 0];
            elseif(j==12)
                skel.tree(1,14).offset = [0, norm(skel.tree(1,14).offset), 0];
                skel.tree(1,17).offset = [0, norm(skel.tree(1,17).offset), 0];
                skel.tree(1,25).offset = [0, norm(skel.tree(1,25).offset), 0];
            elseif(j==13 || j==14) % Head
                skel.tree(1,j+2).offset = [0, norm(skel.tree(1,j+2).offset), 0];
            elseif(j==15) % L SH
                skel.tree(1,18).offset = [0, norm(skel.tree(1,18).offset), 0];
            elseif(j==16) % L UArm
                skel.tree(1,19).offset = [0, norm(skel.tree(1,19).offset), 0];
            elseif(j==17) % L Arm
                skel.tree(1,20).offset = [0, norm(skel.tree(1,20).offset), 0];
            elseif(j==18) % L Thumb
                skel.tree(1,22).offset = [0, 0, norm(skel.tree(1,22).offset)];
            elseif(j==19) % L Hand
                skel.tree(1,23).offset = [0, norm(skel.tree(1,23).offset), 0];
            elseif(j==20) % R SH
                skel.tree(1,26).offset = [0, norm(skel.tree(1,26).offset), 0];
            elseif(j==21) % R UArm
                skel.tree(1,27).offset = [0, norm(skel.tree(1,27).offset), 0];
            elseif(j==22) % R Arm
                skel.tree(1,28).offset = [0, norm(skel.tree(1,28).offset), 0];
            elseif(j==23) % L Thumb
                skel.tree(1,30).offset = [0, 0, norm(skel.tree(1,30).offset)];
            elseif(j==24) % L Hand
                skel.tree(1,31).offset = [0, norm(skel.tree(1,31).offset), 0];
            end
        else
            off = 0;
            for frame=1:size(Positions,1)
                x = [Positions(frame,3*(links(j,:)-1)+1); Positions(frame,3*(links(j,:)-1)+2); Positions(frame,3*(links(j,:)-1)+3)];
                off = off + sqrt(sum((x(:,1)-x(:,2)).^2));
            end
            off = off/size(Positions,1);
%             off = off/10;
            if(j==1 || j==2) %Left leg
                skel.tree(1,j+7).offset = [0, -off, 0];
            elseif(j==3)
                skel.tree(1,j+7).offset = [0, 0, off];
            elseif(j==4)
                skel.tree(1,j+7).offset = [0, 0, off];
            elseif(j==5 || j==6) %Right leg
                skel.tree(1,j-2).offset = [0, -off, 0];
            elseif(j==7) 
                skel.tree(1,j-2).offset = [0, 0, off];
            elseif(j==8)
                skel.tree(1,j-2).offset = [0, 0, off];
            elseif(j==9) % LHips
                skel.tree(1,7).offset = [off, 0, 0];
            elseif(j==10) % RHips
                skel.tree(1,2).offset = [-off, 0, 0];
            elseif(j==11) % Spine
                skel.tree(1,13).offset = [0, off, 0];
            elseif(j==12)
                skel.tree(1,14).offset = [0, off, 0];
                skel.tree(1,17).offset = [0, off, 0];
                skel.tree(1,25).offset = [0, off, 0];
            elseif(j==13 || j==14) % Head
                skel.tree(1,j+2).offset = [0, off, 0];
            elseif(j==15) % L SH
                skel.tree(1,18).offset = [0, off, 0];
            elseif(j==16) % L UArm
                skel.tree(1,19).offset = [0, off, 0];
            elseif(j==17) % L Arm
                skel.tree(1,20).offset = [0, off, 0];
            elseif(j==18) % L Thumb
                skel.tree(1,22).offset = [0, 0, off];
            elseif(j==19) % L Hand
                skel.tree(1,23).offset = [0, off, 0];
            elseif(j==20) % R SH
                skel.tree(1,26).offset = [0, off, 0];
            elseif(j==21) % R UArm
                skel.tree(1,27).offset = [0, off, 0];
            elseif(j==22) % R Arm
                skel.tree(1,28).offset = [0, off, 0];
            elseif(j==23) % L Thumb
                skel.tree(1,30).offset = [0, 0, off];
            elseif(j==24) % L Hand
                skel.tree(1,31).offset = [0, off, 0];
            end
        end
    end
end