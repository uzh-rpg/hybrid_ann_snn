% H36MTrial 
% This is information about data necessary to train one model.
%
classdef H36MTrial
  properties
    train_data;
    test_data;
    val_data;
    trainval_data;
    
    infeat;
  end
  
  methods
    function obj = H36MTrial(train, val, test)
      obj.train_data = train;
      obj.val_data = val;
      obj.test_data = test;
      
      obj.train_data.Sequences = H36MTrial.enumall(train);
      obj.val_data.Sequences = H36MTrial.enumall(val);
      obj.test_data.Sequences = H36MTrial.enumall(test);
			
			obj.trainval_data.Sequences = [obj.train_data.Sequences obj.val_data.Sequences];
    end
    
    function seq = getTrainSequence(obj, n)
      seq = obj.train_data.Sequences(n);
    end
    
    function seq = getValSequence(obj, n)
      seq = obj.val_data.Sequences(n);
		end
		
		function seq = getTrainValSequence(obj, n)
      seq = obj.trainval_data.Sequences(n);
    end
    
    function seq = getTestSequence(obj, n)
      seq = obj.test_data.Sequences(n);
		end
		
		function Ntex = getTrainExampleNo(obj)
			Ntex = 0;
			for i = 1: length(obj.train_data.Sequences)
				Ntex = Ntex + length(obj.train_data.Sequences(i).IdxFrames);
			end
		end
		
		function Ntex = getTestExampleNo(obj)
			Ntex = 0;
			for i = 1: length(obj.test_data.Sequences)
				Ntex = Ntex + length(obj.test_data.Sequences(i).IdxFrames);
			end
		end
		
		function Ntex = getTrainValExampleNo(obj)
			Ntex = 0;
			for i = 1: length(obj.trainval_data.Sequences)
				Ntex = Ntex + length(obj.trainval_data.Sequences(i).IdxFrames);
			end
		end
		
		function Ntex = getValExampleNo(obj)
			Ntex = 0;
			for i = 1: length(obj.train_data.Sequences)
				Ntex = Ntex + length(obj.val_data.Sequences(i).IdxFrames);
			end
		end
		
		function obj = subsampleTrain(obj,Ns)
			for i = 1: length(obj.train_data.Sequences)
				obj.train_data.Sequences(i) = obj.train_data.Sequences(i).subsample(Ns);
			end
			for i = 1: length(obj.train_data.Sequences)
				obj.val_data.Sequences(i) = obj.train_data.Sequences(i).subsample(Ns);
			end
			for i = 1: length(obj.train_data.Sequences)
				obj.trainval_data.Sequences(i) = obj.train_data.Sequences(i).subsample(Ns);
			end
		end
  end
  
  methods(Static)
    function sequence = enumall(data)
      i = 1;
      for s = 1: length(data.subjects)
        for a = 1: length(data.actions)
          for sa = 1: length(data.subactions)
            for c = 1: length(data.cameras)
              sequence(i) = H36MSequence(data.subjects(s), data.actions(a), data.subactions(sa), data.cameras(c));
              
              i = i + 1;
            end
          end
        end
      end
    end
  end
end