classdef H36MFullMask < H36MFeature
  methods
    function obj = H36MFullMask()
      obj.FeatureName = 'FullMask';
      obj.FeaturePath = '';
    end
    
    function frame = process(obj, frame)
    end
    
    function frame = normalize(obj, frame)
    end
    
    function reader  = serializer(obj, Sequence)
			reader = [];
		end
		
		function size		= getFeatureSize(obj)
			size = 0;
		end
  end
end