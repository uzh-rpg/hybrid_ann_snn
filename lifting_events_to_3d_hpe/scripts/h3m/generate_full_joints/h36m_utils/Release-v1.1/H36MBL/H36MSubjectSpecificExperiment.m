classdef H36MSubjectSpecificExperiment < H36MExperiment
    methods
        function obj = H36MSubjectSpecificExperiment()
            db = H36MDataBase.instance();
            
            actions = 2:16;
            cameras = 1:4;
            
            obj.Name = 'SubjectSpecific';
            obj.TestName = obj.Name;
            
            subj = [db.train_subjects db.val_subjects];
            for i = 1: length(subj)
                train.subjects = subj(i);
                train.actions = actions;
                train.subactions = 1;
                train.cameras = cameras;
                
                val.subjects = subj(i);
                val.actions = actions;
                val.subactions = 2;
                val.cameras = cameras;
                
                test.subjects = subj(i);
                test.actions = 1;
                test.subactions = [1 2];
                test.cameras = cameras;
                
                obj.Trials{i} = H36MTrial(train, val, test);
            end
        end
    end
    
    methods(Static)
        function obj = instance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = H36MSubjectSpecificExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end