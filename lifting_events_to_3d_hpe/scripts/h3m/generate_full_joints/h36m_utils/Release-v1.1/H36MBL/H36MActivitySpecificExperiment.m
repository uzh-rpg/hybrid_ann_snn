classdef H36MActivitySpecificExperiment < H36MExperiment
    methods(Access=private)
        function obj = H36MActivitySpecificExperiment()
            db = H36MDataBase.instance();
            
            for a = 2:16
                c = 1:4;
                
                train.subjects = db.train_subjects;
                train.actions = a;
                train.subactions = 1:2;
                train.cameras = c;
                
                val.subjects = db.val_subjects;
                val.actions = a;
                val.subactions = 1:2;
                val.cameras = c;
                
                test.subjects = db.test_subjects(1:3);
                test.actions = a;
                test.subactions = 1:2;
                test.cameras = c;
                
                obj.Trials{a-1} = H36MTrial(train, val, test);
            end
            obj.Name = 'ActivitySpecific';
            obj.TestName = obj.Name;
        end
    end
    
    methods(Static)
        function obj = instance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = H36MActivitySpecificExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end