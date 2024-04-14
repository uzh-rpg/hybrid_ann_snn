classdef H36MGeneralExperiment < H36MExperiment
    methods(Access=private)
        function obj = H36MGeneralExperiment()
            db = H36MDataBase.instance();
            
            obj.Name = 'All';
            obj.TestName = obj.Name;
            
            actions = 2:16;
            cameras = 1:4;
            
            train.subjects		= db.train_subjects;
            train.actions			= actions;
            train.subactions	= [1 2];
            train.cameras			= cameras;
            
            val.subjects			= db.val_subjects;
            val.actions				= actions;
            val.subactions		= [1 2];
            val.cameras				= cameras;
            
            test.subjects			= db.test_subjects(1:3);
            test.actions			= actions;
            test.subactions		= [1 2];
            test.cameras			= cameras;
            
            obj.Trials{1} = H36MTrial(train,val,test);
        end
    end
    
    methods(Static)
        function obj = instance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = H36MGeneralExperiment();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end