classdef H36MRGBVideoFeature < H36MVideoFeature
  properties
    Resize;
    Symmetric;
		Interpolation;
  end
  
  methods
    function obj = H36MRGBVideoFeature(varargin)
      obj.RequiredFeatures = {};
      obj.Symmetric = false;
      
      obj.FeatureName = 'RGB';
      
      obj.Interpolation = 'bicubic';
      obj.FeaturePath = '/Videos/';
      obj.Extension = '.mp4';
			
			for i = 1: 2: length(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
    end
    
    function PFrame = process(obj, Image)
      PFrame = im2uint16(imresize(Image,obj.Resize,'method',obj.Interpolation));
    end
    
    function [NFeat] = normalize(obj,Feat,Subject, Camera)
      if obj.Symmetric
        NFeat = flipdim(Feat,2);
      else
        NFeat = Feat;
      end
    end
    
    function flag = exist(obj, Sequence)
      flag = exist([Sequence.getPath() filesep obj.FeaturePath Sequence.getName obj.Extension],'file');
    end
    
    function reader  = serializer(obj, Sequence)
      reader = H36MVideoDataAccess([Sequence.getPath() filesep obj.FeaturePath Sequence.getName obj.Extension]);
		end  
		
		function 		size		= getFeatureSize(obj)
			% FIXME this needs to be corrected
			size = 0;
		end
  end
end