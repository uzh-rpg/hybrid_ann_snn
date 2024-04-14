classdef H36MMyBGMask < H36MVideoFeature
  properties
    Resize;
    Symmetric;
		scc;
		Perm;
  end
  
  methods
    function obj = H36MMyBGMask(varargin)
      obj.Symmetric = false;
      obj.scc = true;
			obj.Perm = false;
			
			obj = obj.fillin(varargin{:});
			
			if obj.scc
				obj.FeatureName = 'ground_truth_scc';
				obj.FeaturePath = [filesep 'MySegmentsMat' filesep obj.FeatureName(1:12) '_bs' filesep];
			else
				obj.FeatureName = 'ground_truth';
				obj.FeaturePath = [filesep 'MySegmentsMat' filesep obj.FeatureName filesep];
			end
			
      obj.Extension = '.mat';
    end
    
    function FP = process(obj, Frame)
			%we are providing precomputed background subtraction masks so no need
			%for processing here
			Frame = Frame(:,:,1);
    end
    
    function FP = normalize(obj, Frame, ~, ~)
			% pick the largest connected component of the mask ?!
			if obj.scc
				Frame = preproc_mask(Frame, obj.scc);
			end
			
			if obj.Resize ~= 1
				FP = imresize(Frame,obj.Resize,'method','nearest')>0;
			else
				FP = logical(Frame);
			end
			
      if obj.Symmetric
        FP = flipdim(FP,2);
      end
    end
        
    function reader  = serializer(obj, Sequence)
			folder = [Sequence.getPath() filesep obj.FeaturePath obj.FeatureName filesep];
			if ~exist(folder,'dir') && obj.Perm
				mkdir(folder);
			end
			
      reader = H36MFeatureDataAccess([Sequence.getPath() filesep obj.FeaturePath Sequence.getName obj.Extension],'Masks',obj.Perm);        
		end
		
		function 		size		= getFeatureSize(obj,Camera)
			[size(1) size(2)] = Camera.getResolution;
		end
  end
end