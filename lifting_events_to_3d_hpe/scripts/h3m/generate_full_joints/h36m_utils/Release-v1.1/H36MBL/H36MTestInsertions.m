function filename = H36MTestInsertions(experiment, config, trialno, method)

addpaths;

eval(config);

if ~exist('par_args','var')
  error('par_args variable undefined !');
end

FeatTypes = [InputFeatures TargetFeatures];

db = H36MDataBase.instance();

[method  ' ' experiment.Name]

% meta = db.renders;

trialno

% get test sequence
Sequence = experiment.Trials{trialno}.test_data.Sequences;

% compute features
Feats = H36MComputeFeatures(Sequence, InputFeatures);

% load model file
[modeldir modelfilename] = experiment.getModelFileName(FeatTypes, method, tag_base, trialno);
eval(['model = ' method '(' par_args{1} sprintf(',%s',par_args{2:end}), ');']);
model = model.load([modeldir modelfilename '.mat'])

traintime = model.traintime;
testtime = model.testtime;

% does prediction
[Pred{1} model] = model.test(Feats(1:length(InputFeatures)));

% returns to original representation
Pred{1} = TargetFeatures{1}.unnormalize(Pred{1});

% save files in matlab format
[dir filename] = experiment.getResultsFileName(FeatTypes, method, tag, trialno);
save([dir filename '.mat'],'Pred','traintime','testtime');

fprintf('Saved results in : %s\n',[filename '.mat']);
fprintf('Testing finished successfully!\n');
end