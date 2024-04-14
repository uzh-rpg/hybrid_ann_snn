classdef H36MTinyExperiment < H36MExperiment
    methods
        function obj = H36MTinyExperiment()
            db = H36MDataBase.instance();
            
            actions = 13:13;
            cameras = 1:4;
            % 			cameras =1;
            
            obj.Name = 'Tiny';
            obj.TestName = obj.Name;
            
            subj = 1;
            for i = 1: length(subj)
                train.subjects = subj(i);
                train.actions = actions;
                train.subactions = 1;
                train.cameras = cameras;
                
                val.subjects = subj(i);
                val.actions = actions;
                val.subactions = 1;
                val.cameras = cameras;
                
                test.subjects = subj(i);
                test.actions = 13;
                test.subactions = 2;
                test.cameras = cameras;
                
                obj.Trials{i} = H36MTrial(train, val, test);
            end
            db = db.setExperiment(obj);
        end
    end
    
    methods(Static)
        function obj = instance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = H36MTinyExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end