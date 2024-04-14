function H36MTrainModel(experiment, config, trialno, method, val)
% Function to train one of the models
% - configfile 
% - trialno (trial number)
% - method 
% - is it a validation run ?

addpaths;

eval(config);


if ~exist('par_args','var')
  error('par_args variable undefined !');
end

fprintf('Training trial number %d\n',trialno);

FeatTypes = [InputFeatures, TargetFeatures];
Nif = length(InputFeatures);
Ntf = length(TargetFeatures);

trial = experiment.Trials{trialno};

if val
	Ntex = trial.getTrainExampleNo();
	sequences = trial.train_data.Sequences;
else
	if exist('val_vals','var')
		for nval = 1: length(val_vals)
			tag_cv = ['__val_' num2str(nval)];
			[directory filename] = experiment.getResultsFileName(FeatTypes, method, [tag_base tag_cv], trialno);
			load([directory filename '.mat'],'ValError','par_args');
			ValErr(nval) = ValError;
		end
		[~, ind] = min(ValErr);
		disp( ['Best Parameter value ' val_arg ' = ' num2str(val_vals(ind))])
		par_args = [par_args {val_arg, 'val_vals(ind)'}];
	end
	Ntex = trial.getTrainValExampleNo();
	sequences = trial.trainval_data.Sequences;
end

t = tic;
f = 0;
for i = 1: length(sequences)
  sequence = sequences(i);
  
  Feats = H36MComputeFeatures(sequence,FeatTypes);
  
  if i == 1 && length(par_args) >= 1
    eval(['model = ' method '(' par_args{1} sprintf(',%s',par_args{2:end}), ',''Ntex'',' num2str(Ntex) ')']);
		dispprogress('Progress ',0,Ntex,toc(t));
  elseif i == 1
    eval(['model = ' method '(''Ntex'',' num2str(Ntex) ')']);
		dispprogress('Progress ',0,Ntex,toc(t));
	end
  
	% map
  model = model.update(Feats(1:Nif), Feats(Nif+1:Nif+Ntf));
	
	f = sequence.getNumFrames() + f;
	dispprogress('Progress ',f,Ntex,toc(t));
end 

% train operation
% reduce
model = model.train();

if val
	[directory filename] = experiment.getModelFileName(FeatTypes, method, tag, trialno);
	if ~exist(directory,'dir')
		mkdir(directory);
	end
	model.save([directory filename '.mat']);
else
	[directory filename] = experiment.getModelFileName(FeatTypes, method, tag_base, trialno);
	if exist('ValErr','var')
		model.save([directory filename '.mat']);
		save([directory filename '.mat'],'-append','ValErr','val_vals','val_arg');
		% FIXME delete the validation files
	else
		model.save([directory filename '.mat']);
	end
end
fprintf('Finished training %s%s\n',directory,filename);
end