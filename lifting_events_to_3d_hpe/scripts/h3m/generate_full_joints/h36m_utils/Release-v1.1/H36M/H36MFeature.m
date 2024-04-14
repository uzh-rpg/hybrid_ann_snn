classdef H36MFeature
  properties (SetAccess = protected, GetAccess = public)
    FeatureName;
    FeaturePath;
    FrameMultiplier;
    Extension;
    RequiredFeatures;
  end
    
  methods
    function obj = fillin(obj, varargin)
      if mod(length(varargin),2) ~= 0
        error('Wrong number of parameters!');
      end
      
      for i = 1: 2: length(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
    end
    
    function flag = exist(obj, Sequence)
      flag = exist([Sequence.getPath() filesep obj.FeaturePath Sequence.getName '__' obj.FeatureName obj.Extension],'file');
		end
		
		
	end
end