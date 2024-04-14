db = H36MDataBase.instance();


InputFeatures{1} = H36MPose3DPositionsFeature('Monocular',true,'Relative',true,'Part','body');
TargetFeatures{1} = H36MPose3DPositionsFeature('Monocular',true,'Relative',true,'Part','body');

InputFeatures{1}.FeatureName
TargetFeatures{1}.FeatureName

s = RandStream.create('mt19937ar','seed',5489);
RandStream.setDefaultStream(s);

IKParam = 1.7;
switch method
	case 'H36MMeanTarget'
		par_args = {};
	case 'none'
	case 'H36MBLKnn'
		par_args = {'''dist''','''euclidean''','''method''','''mean'''};
		
	case 'H36MBLKr'
		par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',num2str(IKParam),'''MAX_SAMPLES''','40000'};

	case 'H36MBLLinKrr'
		warning('50K dim linkrr');
		par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',num2str(IKParam),'''lambda''','2200','''Napp''','40000'};
	
	case 'H36MBLLinKrrVedaldi'
		par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',num2str(IKParam),'''lambda''','1000','''Napp''','15000'};
	
	case 'H36MBLLinKrrNoBias'
		nval = str2num(getenv('OKPAR'));
		if ~isempty(nval)
			par_arg = '''lambda''';
			par_vals = logspace(-6,6,13);%[1e-2 1e-1 1 5 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5 5e5 1e6 1e-3 1e-4 1e-5 1e-6];
			par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',...
				num2str(IKParam),'''MAX_SAMPLES''','40000','''lambda''','1000','''Napp''','15000',...
				par_arg,'par_vals(nval)'};
			tag_base = [tag_base num2str(nval)];
		else
			par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',...
				num2str(IKParam),'''MAX_SAMPLES''','40000','''lambda''','1000','''Napp''','10000'};
		end	
		
	case 'H36MBLKde'
		%[0.05 - 0.5]
		par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',num2str(IKParam),'''MAX_SAMPLES''','10000','''Napp_x''','5000','''OKParam''','.2','''Napp_y''','2000','''lambda''','1000'};
		
	case 'H36MBLKde2'
		par_args = {'''dim''','size(Feats{1},2)','''IKType''','''exp_chi2''','''IKParam''',num2str(IKParam),'''MAX_SAMPLES''','10000'};
		
	otherwise
		error('Unknown method!');
end

tag_base = 'alpha_test';
if exist('tag_cv','var')
	tag = [tag_base tag_cv];
else
	tag = tag_base ;
end

tag
