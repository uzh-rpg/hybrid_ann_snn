classdef H36MExperiment
    properties
        Trials;
        Name;
        TestName;
        data_dir;
    end
    
    methods
        function obj = H36MExperiment()
            db = H36MDataBase.instance();
            obj.data_dir = db.exp_dir;
        end
        
        function Trial = getTrial(obj, i)
            Trial = obj.Trials{i};
        end
        
        function [modeldir name] = getFeatureModelFileName(obj, Features, method, tag, trialno)
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                modeldir = [db.model_dir 'MyModels' filesep obj.Name filesep 'none_' Features{1}.FeatureName filesep];
            else
                modeldir = [db.model_dir 'MyModels' filesep obj.Name filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName filesep];
            end
            if ~exist(modeldir,'dir')
                mkdir(modeldir);
            end
            name = [method '__' tag '__trial_' sprintf('%04d',trialno) ];
        end
        
        function [modeldir name] = getFeatureResultsFileName(obj, Features, method, tag, trialno)
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                modeldir = [db.model_dir 'MyResults' filesep obj.Name filesep 'none_' Features{1}.FeatureName filesep];
            else
                modeldir = [db.model_dir 'MyResults' filesep obj.Name filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName filesep];
            end
            if ~exist(modeldir,'dir')
                mkdir(modeldir);
            end
            name = [method '__' tag '__trial_' sprintf('%04d',trialno) ];
        end
        
        function [modeldir name] = getModelFileName(obj, Features, method, tag, trialno)
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                modeldir = [db.model_dir 'MyModels' filesep obj.Name filesep 'none_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            else
                modeldir = [db.model_dir 'MyModels' filesep obj.Name filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            end
            if ~exist(modeldir,'dir')
                mkdir(modeldir);
            end
            name = [method '__' tag '__trial_' sprintf('%04d',trialno) ];
        end
        
        function [resultsdir name] = getResultsFileName(obj, Features, method, tag, trialno)
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                resultsdir = [db.model_dir 'MyResults' filesep obj.Name filesep 'none_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            else
                resultsdir = [db.model_dir 'MyResults' filesep obj.Name filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            end
            if ~exist(resultsdir,'dir')
                mkdir(resultsdir);
            end
            name = [method '__' tag '__trial_' sprintf('%04d',trialno)];
        end
    end
end