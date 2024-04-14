classdef H36MMyBBMask < H36MVideoFeature
  properties
    Resize;
    Symmetric;
		Perm;
	end
	
  methods
    function obj = H36MMyBBMask(varargin)
			obj.FeatureName = 'ground_truth_bb';
			obj.FeaturePath = ['/MySegmentsMat' filesep obj.FeatureName filesep];
			obj.Perm = true;
      obj.Symmetric = false;
      
      for i = 1: 2: length(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
      
      obj.Extension = '.mat';
			obj.RequiredFeatures{1} = H36MPose3DPositionsFeature();
    end
    
    function Frame = process(obj, Frame, ~, Camera)
			% 
			N = 14 * 32;
			Frame2 = zeros(N,3);
			
            %%
			r = sqrt(2)/2;
			sphere = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1; r r r; r r -r; r -r r; r -r -r; -r r r; -r r -r; -r -r r; -r -r -r];

			for l = 1: length(Frame)/3
                radius = 115;
                if(l==15 || l==16), radius = 140; end;
                if(l==12 || l==13 || l==14), radius = 260; end;
				Frame2((l-1)*14+1:l*14,:) = sphere*radius+repmat(Frame((l-1)*3+1:3*l),[14 1]);
			end
			
			FP = Camera.project(Frame2);
			FP2D = reshape(FP,[2,N])';
			m = floor(min(FP2D,[],1)); M = ceil(max(FP2D,[],1)); 
			[W H] = Camera.getResolution;

            %Plotting point cloud used
            %plot(FP2D(:,1),FP2D(:,2),'r*'),% hold on;

            %Clipping to screen coordinates
            m = max(m,1);
            M(1) = min(M(1), W);
            M(2) = min(M(2), H);
            
			Frame = false(W,H);
			Frame(m(2):M(2),m(1):M(1)) = true;
    end
    
    function Frame = normalize(obj, Frame, ~, ~)
			if ~isempty(obj.Resize)
				Frame = imresize(Frame,obj.Resize,'method','nearest')>0;
			end
			
      if obj.Symmetric
        Frame = flipdim(Frame,2);
      end
    end
    
    function flag = exist(obj, Sequence)
      flag = exist([Sequence.getPath() filesep obj.FeaturePath Sequence.getName obj.Extension],'file');
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