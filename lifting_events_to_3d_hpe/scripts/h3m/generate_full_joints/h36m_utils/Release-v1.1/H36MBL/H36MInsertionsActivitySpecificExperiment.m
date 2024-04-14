classdef H36MInsertionsActivitySpecificExperiment < H36MExperiment
    methods(Access=private)
        function obj = H36MInsertionsActivitySpecificExperiment()
            db = H36MDataBase.instance();
            
            meta = db.renders;
            
            for a = 1:7
                c = 1:4;
                
                train.subjects = db.train_subjects;
                train.actions = meta.actions(a)-1;
                train.subactions = 1:2;
                train.cameras = c;
                
                val.subjects = db.val_subjects;
                val.actions = meta.actions(a)-1;
                val.subactions = 1:2;
                val.cameras = c;
                
                test.subjects = 0;
                test.actions = a;
                test.subactions = 1;
                test.cameras = 1;
                
                obj.Trials{a} = H36MTrial(train, val, test);
            end
            obj.Name = 'InsertionsActivitySpecific';
            obj.TestName = obj.Name;
        end
    end
    
    methods
        function [modeldir name] = getModelFileName(obj, Features, method, tag, trialno)
            db = H36MDataBase.instance();
            if ~any(strcmp(properties(Features{1}), 'MaskFeature'))
                modeldir = [db.model_dir 'MyModels' filesep 'ActivitySpecific' filesep 'none_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            else
                modeldir = [db.model_dir 'MyModels' filesep  'ActivitySpecific' filesep Features{1}.MaskFeature.FeatureName '_' Features{1}.FeatureName '__' Features{2}.FeatureName filesep];
            end
            if ~exist(modeldir,'dir')
                mkdir(modeldir);
            end
            name = [method '__' tag '__trial_' sprintf('%04d',obj.Trials{trialno}.train_data.actions) ];
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
                obj = H36MInsertionsActivitySpecificExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end