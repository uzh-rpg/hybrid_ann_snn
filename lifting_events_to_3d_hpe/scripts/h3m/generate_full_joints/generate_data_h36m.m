
dataset = H36MDataBase.instance();

Features{1} = H36MPose3DPositionsFeature();


for s = [5 6 7 8 9 11 1]
    for a = 2:16
        for b = 1:2
            for c = 1:4
            tt = tic;
            fprintf('  subject %02d, action %02d-%d',s,a,b);
            path = sprintf('%s/S%01d/MyPoseFeatures/FULL_D3_Positions/', dataset.exp_dir, s); 
            
            makedir(path)
            Sequence = H36MSequence(s, a, b, c);
            fprintf(Sequence.Name);
            F = H36MComputeFeatures(Sequence, Features);
            if iscell(F)
            save(sprintf('%s%s.mat', path, Sequence.BaseName), 'F');
            end
            end
        end
    end
end
