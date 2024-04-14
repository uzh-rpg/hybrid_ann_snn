function Err = H36MTestModel(experiment, config, trialno, method, val)
% Function to test one of the models
% - configfile 
% - trialno
% - method 
% - is it a validation run ?
addpaths;

eval(config);


if ~exist('par_args','var')
  error('par_args variable undefined !');
end

fprintf('Testing trial number %d\n',trialno);
trial = experiment.Trials{trialno};


if val 
	subsample = 20;
  Sequences = trial.val_data.Sequences;
  FeatTypes = [InputFeatures TargetFeatures];
	[modeldir modelfilename] = experiment.getModelFileName(FeatTypes, method, tag, trialno);
	Ntex = trial.getTestExampleNo()/subsample;
else
  Sequences = trial.test_data.Sequences;
	FeatTypes = InputFeatures;
	[modeldir modelfilename] = experiment.getModelFileName( [InputFeatures TargetFeatures], method, tag_base, trialno);
	Ntex = trial.getTestExampleNo();
end

t = tic;
f = 0;

for i = 1: length(Sequences)
  % test model on chunk
  % map
  sequence = Sequences(i);
  
	if val
		% speed up validation especially useful for KDE
		sequence = sequence.subsample(subsample);
	end
	
	Feats = H36MComputeFeatures(sequence, FeatTypes);
	
	if i==1 && length(par_args) > 1
		eval(['model = ' method '(' par_args{1} sprintf(',%s',par_args{2:end}),  ',''Ntex'',' num2str(Ntex) ');']);
		model = model.load([modeldir modelfilename '.mat'])
		if ~val, dispprogress('Progress ',0,Ntex,toc(t)); end;
	elseif i==1
		eval(['model = ' method '();']);
		model = model.load([modeldir modelfilename '.mat'])
		if ~val, dispprogress('Progress ',0,Ntex,toc(t)); end;
	end

  [Pred{i} model] = model.test(Feats(1:length(InputFeatures)));
  
	Pred{i} = TargetFeatures{1}.unnormalize(Pred{i});
	
  if val  
		Feats{2} = TargetFeatures{1}.unnormalize(Feats{2});
		if strcmp(TargetFeatures{1}.Type, TargetFeatures{1}.POSITIONS_TYPE)
			[Err{i} Errm{i}] = H36MError('mpjpd', Feats{2}, Pred{i}, TargetFeatures{1}, sequence.getSubject);
		else
			[Err{i} Errm{i}] = H36MError('mpjad', Feats{2}, Pred{i}, TargetFeatures{1}, sequence.getSubject);
		end
		
    NEx(i) = sequence.getNumFrames;
		fprintf('Partial error : %.02f\n', mean(Err{i}));
	else
		
		f = sequence.getNumFrames() + f;
		dispprogress('Progress ',f,Ntex,toc(t));
	end
	
end

dispprogress('Progress ',Ntex,Ntex,toc(t));
fprintf('\n\n\n');

fprintf('Train time : %.02f\n',model.traintime);
fprintf('Test time  : %.02f\n',model.testtime);

traintime = model.traintime;
testtime = model.testtime;

fprintf('Finished testing in %f\n',testtime);

if val
	ValError = mean(cell2mat(Err'));
  fprintf('Final error = %.02f\n', ValError);
	
	[dir filename] = experiment.getResultsFileName(FeatTypes, method, tag, trialno);
	if ~exist(dir,'dir')
		mkdir(dir);
	end
	save([dir filename '.mat'],'Pred','traintime','testtime','Err','NEx','Errm','ValError','val_vals');
else
	[dir filename] = experiment.getResultsFileName([InputFeatures TargetFeatures], method, tag_base, trialno);
	save([dir filename '.mat'],'Pred','traintime','testtime');
  disp('Nothing to compute since test data unavailable !');
	Err = NaN;
end

fprintf('Saved results in : %s\n',[filename '.mat']);
fprintf('Testing finished successfully!\n');
end
