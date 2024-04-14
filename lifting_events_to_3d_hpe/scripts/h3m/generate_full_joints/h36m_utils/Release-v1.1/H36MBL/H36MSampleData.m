function H36MSampleData(experiment, config, trialno, sampling_method, featind,MAX_SAMPLES)

addpaths;
method = 'none';

db = H36MDataBase.instance();
directory = [db.model_dir 'MySampling/' experiment.Name '/'];
filename =  [sampling_method   '_' sprintf('%02d',trialno)];
if exist([directory filename '.mat'],'file')
	return;
end

eval(config);



fprintf('Training trial number %d\n',trialno);

FeatTypes = [InputFeatures, TargetFeatures];
FeatTypes = FeatTypes(featind);

trial = experiment.Trials{trialno};

Ntex = trial.getTrainValExampleNo();
sequences = trial.trainval_data.Sequences;

t = tic;
f = 0;
if Ntex > MAX_SAMPLES
	sample_rate = ceil(Ntex/MAX_SAMPLES);
else
	sample_rate = 1;
end
FeatSample = cell(1,length(FeatTypes));
for i = 1: length(sequences)
  sequence = sequences(i);
  
  Feats = H36MComputeFeatures(sequence,FeatTypes);
  
	switch sampling_method
		case 'uniform'
			% map
			for ff = 1: length(FeatTypes)
				FeatSample{ff} = [FeatSample{ff}; Feats{ff}(1:sample_rate:end,:)];
			end
			
	end
	
	f = sequence.getNumFrames() + f;
	dispprogress('Progress ',f,Ntex,toc(t));
end

% train operation
% reduce

db = H36MDataBase.instance();
directory = [db.model_dir 'MySampling/' experiment.Name '/'];
filename =  [sampling_method   '_' sprintf('%02d',trialno)];
if ~exist(directory,'dir')
	mkdir(directory);
end
save([directory filename '.mat'],'FeatSample');

fprintf('Finished training %s%s\n',directory,filename);
end