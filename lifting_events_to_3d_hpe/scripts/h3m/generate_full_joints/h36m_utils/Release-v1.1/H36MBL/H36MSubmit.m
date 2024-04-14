function ofilename = H36MSubmit(experiment, config, trialno, method, ofilename)
% H36MSubmit generates the result files that are accepted by our server.
% FIXME add variable dimensionality for the predictions
eval(config);

% save 
FeatTypes = [InputFeatures TargetFeatures];
[directory resfile] = experiment.getResultsFileName(FeatTypes, method, tag_base, trialno);

if nargin < 5
    load([directory resfile '.mat']);
	ofilename = [directory resfile '.results'];
else
    load([ofilename sprintf('__trial_%04d',trialno) '.mat']);
	directory = [];
	ofilename = [ofilename sprintf('__trial_%04d',trialno) '.results'];
end
hf = fopen( ofilename,'w+');
trials = experiment.Trials;

% experiment name
fwrite(hf,length(experiment.TestName),'uint8');
fwrite(hf,experiment.TestName,'char');

% feature type
fwrite(hf,length(TargetFeatures{1}.FeatureName),'uint8');
fwrite(hf,TargetFeatures{1}.FeatureName,'char');
fwrite(hf,TargetFeatures{1}.Relative,'uint8');
if isempty(TargetFeatures{1}.Part)
	fwrite(hf,0,'uint8');
elseif strcmp(TargetFeatures{1}.Part,'body')
	fwrite(hf,1,'uint8');
else
	error('Invalid Part! Only full skeleton or reduced ''body'' skeleton are allowed in submissions!');
end

% number of sequences
sequences = trials{trialno}.test_data.Sequences;

if length(sequences)~=length(sequences)
	error('Problem!');
end

fwrite(hf,uint16(length(sequences)),'uint16');
fprintf(1,'Sequences: %03d | %03d\n',0,length(sequences)); 

for s = 1: length(sequences)
	fprintf(1,'\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bSequences: %03d | %03d\n',s,length(sequences)); 

	% name of sequence
	fwrite(hf,length(sequences(s).Name),'uint8'); fwrite(hf,sequences(s).Name,'char');
	[N D] = size(Pred{s});
	fwrite(hf,N,'uint16'); fwrite(hf,D,'uint16');
	for i = 1: N
		fwrite(hf,single(Pred{s}(i,:)),'float');
	end
end
fclose(hf);

disp(['Saved results to file ' ofilename]);

% connect and send file

end