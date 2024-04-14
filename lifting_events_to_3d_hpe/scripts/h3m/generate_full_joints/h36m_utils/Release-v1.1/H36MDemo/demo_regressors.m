%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this code example shows how to train, test and get a result file for
%% a simple regressor
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpaths;
% clear;
if ~exist('experiment','var')
% 	experiment = H36MSubjectSpecificExperiment.instance();
	experiment = H36MTinyExperiment.instance();
end

if ~exist('configfile','var')
	configfile = 'H36MBasicConfig';
end

if ~exist('method','var')
% 	method = 'H36MBLLinKrr';
% 	method = 'H36MBLKr';
	method = 'H36MBLKnn';
end

if ~exist('val','var')
	val = false;
end

resfile = 'results.txt';

method

for i = 1
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	disp('2) train a small model');
	H36MTrainModel(experiment, configfile, i, method, val);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	disp('3) test model');
	H36MTestModel(experiment, configfile, i,method, val);
	if ~val
		filename = H36MSubmit(experiment,configfile, i, method);
		H36MComputeError(filename, i, resfile);
	end
end

