classdef H36MInsertionsGeneralExperiment < H36MExperiment
    methods(Access=private)
        function obj = H36MInsertionsGeneralExperiment()
            db = H36MDataBase.instance();
            
            obj.Name = 'InsertionsAll';
            obj.TestName = obj.Name;
            
            actions = 2:6;
            cameras = 1:4;
            
            train.subjects		= db.train_subjects;
            train.actions			= actions;
            train.subactions	= [1 2];
            train.cameras			= cameras;
            
            val.subjects			= db.val_subjects;
            val.actions				= actions;
            val.subactions		= [1 2];
            val.cameras				= cameras;
            
            for i = 1: 7
                test.subjects			= 0;
                test.actions			= i;
                test.subactions		= 1;
                test.cameras			= 1;
                
                obj.Trials{i} = H36MTrial(train,val,test);
            end
        end
    end
    
    methods
        function [modeldir name] = getModelFileName(obj, Features, method, tag, ~)
            trialno = 1;
            
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                modeldir = [db.model_dir 'MyModels' filesep obj.Name filesep 'none_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            else
                modeldir = [db.model_dir 'MyModels' filesep 'All' filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
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
            name = [method '__' tag '__insertion_' sprintf('%04d',trialno) ];
        end
    end
    
    methods(Static)
        function obj = instance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = H36MInsertionsGeneralExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end