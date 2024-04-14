function [Exp, Err, trialno, error_measure, seq_num] = H36MComputeError(file, trialno, outfile, error_measure, locked)
Exp = '';
Err = '';
seq_num = 0;

if(exist('error_measure','var') && isempty(error_measure))
    clear error_measure;
end

disp('=============================================');
disp('===========   Computing Errors   ============');
disp('=============================================');

if ~exist('locked','var')
    locked = true;
end

hf = fopen(file);

% experiment name
n = fread(hf,1,'uint8');
Exp = fread(hf,n,'*char');
switch Exp'
    case 'Tiny'
        experiment = H36MTinyExperiment.instance();
    case 'ActivitySpecific'
        experiment = H36MActivitySpecificExperiment.instance();
    case 'ActivitySpecificFull'
        experiment = H36MActivitySpecificFullExperiment.instance();
    case 'SubjectSpecific'
        experiment = H36MSubjectSpecificExperiment.instance();
    case 'All'
        experiment = H36MGeneralExperiment.instance();
    case 'AllFull'
        experiment = H36MGeneralFullExperiment.instance();
    case 'InsertionsActivitySpecific'
        experiment = H36MInsertionsActivitySpecificExperiment.instance();
    case 'InsertionsAll'
        experiment = H36MInsertionsGeneralExperiment.instance();
    otherwise
        saveerror(outfile,sprintf('Error parsing file. Expected experiment name found "%s"\n',Exp'));
        return;
end

n = fread(hf,1,'uint8'); featname = fread(hf,n,'*char')'; relative = fread(hf,1,'uint8')';
p = fread(hf,1,'uint8')';
if p == 0
    Part = '';
else
    Part = 'body';
end

switch featname
    case 'D3_Positions_mono_universal'
        PredictionFeature = H36MPoseUniversal3DPositionsFeature('Monocular',true,'Relative',relative,'Part',Part);
        if ~exist('error_measure','var')
            error_measure = 'umpjpd';
        end
    case 'D3_Positions_mono'
        PredictionFeature = H36MPose3DPositionsFeature('Monocular',true,'Relative',relative,'Part',Part);
        if ~exist('error_measure','var')
            error_measure = 'mpjpd';
        end
    case 'D3_Angles_mono'
        PredictionFeature = H36MPose3DAnglesFeature('Monocular',true,'Relative',relative,'Part',Part);
        if ~exist('error_measure','var')
            error_measure = 'mpjad';
        end
    case 'D2_Positions'
        PredictionFeature = H36MPose2DPositionsFeature('Part',Part);
        if ~exist('error_measure','var')
            error_measure = 'mpjpd';
        end
    otherwise
        saveerror(outfile,sprintf('Error parsing file. Expected feature name found "%s"\n',featname));
        return;
end

switch error_measure
    case {'umpjpd'}
        TargetFeature = H36MPoseUniversal3DPositionsFeature('Monocular',true,'Relative',relative,'Part','body');
    case {'mpjpd'}
        TargetFeature = H36MPose3DPositionsFeature('Monocular',true,'Relative',relative,'Part','body');
    case {'mpjad','mpjad-nogr'}
        TargetFeature = H36MPose3DAnglesFeature('Monocular',true,'Relative',relative,'Part','body');
    case {'l2'}
        TargetFeature = PredictionFeature;
end

experiment.Name

trials = experiment.Trials;

sequences = trials{trialno}.test_data.Sequences;
seq_num = length(sequences);

ns = fread(hf,1,'uint16');
if ns ~= length(sequences)
    saveerror(outfile,sprintf('Error parsing file. Trial %d expected number for sequences "%d" found "%d"\n',trialno,length(sequences),ns));
    return;
end

if locked
    % get the lock on the file (for parallel environments)
    h = true;
    while h
        pause(1);
        disp('waiting');
        h = exist([outfile '.lock'],'file');
    end
    system(['touch ' outfile '.lock']);
end

if ~exist([outfile '.mat'],'file')
    Err = NaN(length(trials),1);
    Accuracy = cell(length(trials),1);
    AccuracyPJ = cell(length(trials),1);
else
    load([outfile '.mat']);
end
% save an intermediate thing with all the results and update it here

for s = 1: ns
    disp(s/ns*100);
    
    % save sequence
    tfs = PredictionFeature.serializer(sequences(s));
    
    % name of sequence
    n = fread(hf,1,'uint8'); name = fread(hf,n,'*char')';
    if ~strcmp(sequences(s).Name,name)
        saveerror(outfile, sprintf('Error parsing file. Expected "%s" but "%s" found!',sequences(s).Name,name));
        return;
    end
    
    Subject = sequences(s).getSubject;
    Camera = sequences(s).getCamera;
    
    N = fread(hf,1,'uint16'); D = fread(hf,1,'uint16');
    Pred = zeros(N,D);
    for i = 1: N
        Pred(i,:) = fread(hf,D,'float');
    end
    
    Fn = H36MComputeFeatures(sequences(s),{TargetFeature});
    
    [E{s} Epj{s}] = H36MError(error_measure, Fn{1}, Pred, PredictionFeature, Subject);
end

switch error_measure
    case {'mpjpd','mpjad','mpjad-nogr','umpjpd','l2'}
        Err(trialno) = mean(cell2mat(E'));
end

switch error_measure
    case {'mpjpd','umpjpd'}
        EE{trialno} = cell2mat(Epj');
        distances = 5:5:200;
        AccuracyPJ{trialno} = zeros(length(distances),size(EE{trialno},2));
        for i = 1: length(distances)
            AccuracyPJ{trialno}(i,:) = mean(EE{trialno}<distances(i));
            Accuracy{trialno}(i) = mean(AccuracyPJ{trialno}(i,:));
        end
        Accuracy{trialno}
    otherwise
        Accuracy = [];
        AccuracyPJ = [];
        distances = [];
end

disp('=========== Error ===========');
fprintf(1,'Final result %f\n',Err(trialno));
disp('=============================');

fclose(hf);

save([outfile '.mat'],'Err','Accuracy','AccuracyPJ','distances');

%% writing out the results
% modify here for better output format
hf = fopen(outfile,'w+');
fprintf(hf,'%f\t',Err);
fclose(hf);

disp('======= File Contents ========');
system(['cat ' outfile]);disp(' ');
disp('==============================');

if locked
    % remove lock
    if(exist([outfile '.lock'], 'file'))
        delete([outfile '.lock']);
    end
end

end

function saveerror(outfile,message)
hf = fopen(outfile,'w+');
fprintf(hf,'%s\n',message);
fclose(hf);
end