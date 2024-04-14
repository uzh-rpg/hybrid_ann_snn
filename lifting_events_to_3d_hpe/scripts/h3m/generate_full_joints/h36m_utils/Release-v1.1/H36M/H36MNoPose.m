classdef H36MNoPose < H36MPoseFeature
  methods
    function obj = H36MNoPose()
      obj.FeatureName = 'NoPose';
    end
    
    function Frame = process(obj, Frame, Subject, Camera)
		end
    
    function F = normalize(obj, F, Subject, Camera)
		end
		
		function     reader  = serializer(obj, Sequence)
		end
		
		function Feat   = unnormalize(obj,Feat)
		end
		
		function Pos = toPositions(obj, Data, skel)
			Pos = [];
		end
		
		function [NFeat skel2] = select(obj, Feat, skel, part)
		end
		
		function size		= getFeatureSize(obj)
			size = 0;
		end
  end
end