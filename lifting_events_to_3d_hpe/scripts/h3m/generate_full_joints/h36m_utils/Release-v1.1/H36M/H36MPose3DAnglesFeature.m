classdef H36MPose3DAnglesFeature < H36MPoseFeature
	properties
		Transform;
	end
	
	methods
		function obj    = H36MPose3DAnglesFeature(varargin)
      obj.Dimensions= 3;
      obj.Type			= obj.ANGLES_TYPE;
			obj.Part			= [];
      obj.Relative	= false;
      obj.Monocular = false;
      obj.Symmetric = false;
			obj.Serialize = true;
      obj.Transform = 'deg2360';
      obj = obj.fillin(varargin{:});
      
      obj.FeaturePath = '/MyPoseFeatures/';
			
			if obj.Monocular
				mono = '_mono';
			else
				mono = '';
			end
			
	    obj.FeatureName = sprintf('D%d_%s%s',obj.Dimensions,obj.Type,mono);
			
%       obj.Extension = '.joints.mat';
			obj.Extension = '.cdf';
			
			obj.RequiredFeatures{1} = H36MRawPoseFeature();
		end
		
		function NFeat			= process(obj, Feat, ~, Camera)
			NFeat							= TransformAngles2BVH(Feat(:,4:end),1:size(Feat,1),Feat(:,1:3));
			if obj.Monocular
				T               = (NFeat(:,1:3)-Camera.T)*Camera.R';
        NFeat           = [T TransformJointsAngles( NFeat(:,4:6), NFeat(:,1:3), Camera.T, Camera.R ) NFeat(:,7:end)];  
			end
		end
		
		function [NFeat	skel]	= normalize(obj, Feat, Subject, ~)
			NFeat = Feat;
			
			if obj.Symmetric 
				%FIXME warning('Mirror symmetric angles are not implemented yet!');
			end
			
			if obj.Relative
				% set translation to 0
				NFeat(:,1:obj.Dimensions) = 0; 
			end
			
			% select a subset of angles if necessary
			if ~isempty(obj.Part)
				[NFeat skel] = obj.select(NFeat, Subject.getAnglesSkel, obj.Part);
			else
				skel = Subject.getPosSkel;
			end
			
			% transform to the desired representation
			NAngles = normalize_angles(NFeat(:,4:end),obj.Transform);
			
			% put translation and angles together
			NFeat = [NFeat(:,1:3) NAngles];
		end
				
		function NFeat   = unnormalize(obj,Feat)
			switch obj.Transform
				case 'sin_cos'
					NAngles = normalize_angles(Feat(:,4:end),'atan_sin_cos');	
					NFeat = [Feat(:,1:3) NAngles];
					
				otherwise
					NFeat = Feat;
			end
		end
		
		function Pos		= toPositions(obj, Data, skel)
			N = size(Data,1);
			for i = 1: N
				Pos2 = skel2xyz(skel, Data(i,:))';
				Pos(i,:)  = Pos2(:)';
			end
		end
		
		function [NFeat skel2] = select(obj, Feat, skel, part)		
			if isempty(Feat)
				% if we just want the skeleton
				Feat=zeros(1,78);
			end
			
      switch part
        case 'rootpos'
          joints = 1;
        case 'rootrot'
          joints = 1;
        case 'leftarm'
          joints = 17:24;% p/p2/a fine
        case 'rightarm'
          joints = 25:32;% p/p2/a fine
        case 'head'
          joints = 14:16;% p/p2/a fine
        case 'rightleg'
          joints = 2:6;% p/p2/a fine
        case 'leftleg'
          joints = 7:11;% p/p2/a fine
        case 'upperbody'
          joints = [14:32];% p/p2/a fine
        case 'arms'
          joints = [16:32];% p/p2/a fine
        case 'legs'
          joints = 1:11;% p/p2/a fine
        case 'body'
          joints = [1 2 3 4 7 8 9 12 13 14 15 16 17 18 19 20 25 26 27 28];% p/p2/a fine
				case 'full'
          joints = 1:32;% p/p2/a fine
        otherwise
          error('Unknown');
      end
      
      skel2 = skel;
			skel2.tree = skel.tree(1);
      p = 1;
			joints2 = [];
      for i = 1: length(joints)
        % take node corresponding to joint
        skel2.tree(i) = skel.tree(joints(i));
        skel2.tree(i).children = [];
        
        % update the channels
				if i == 1
					p = p + 3;
				end
				
        % update parents and children
				skel2.tree(i).parent = find(skel.tree(joints(i)).parent==joints);
				
        for j = 1: length(skel.tree(joints(i)).children)
          a = find(skel.tree(joints(i)).children(j)==joints);
          if ~isempty(a)
            skel2.tree(i).children = [skel2.tree(i).children a];
						if ~any(find(joints(i)==joints2))
							joints2 = [joints2 joints(i)];
							skel2.tree(i).rotInd = p:p+length(skel.tree(joints(i)).rotInd)-1;%[p+length(skel.tree(joints(i)).rotInd)-2 p+length(skel.tree(joints(i)).rotInd)-1 p];
							p = p + length(skel.tree(joints(i)).rotInd);
						end
					else
						skel2.tree(i).rotInd = [];
          end
        end
        if isempty(skel2.tree(i).parent)
          skel2.tree(i).parent = 0;
        end
      end
      idx = [skel.tree(1).posInd skel.tree(joints2).rotInd];
      
      NFeat = Feat(:,idx);
		end
		
		function flag = exist(obj, Sequence)
      flag = exist([Sequence.getPath() filesep obj.FeaturePath obj.FeatureName filesep Sequence.getName() obj.Extension],'file');
    end
    
    function reader  = serializer(obj, Sequence)
			folder = [Sequence.getPath() filesep obj.FeaturePath obj.FeatureName filesep];
			if ~exist(folder,'dir') && obj.Serialize
				mkdir(folder);
			end
			
			if obj.Monocular
				reader = H36MFeatureDataAccess([folder Sequence.getName obj.Extension],'Pose', true);
			else
				reader = H36MFeatureDataAccess([folder Sequence.getBaseName obj.Extension],'Pose', true);
			end
		end
		
		function 		size		= getFeatureSize(obj)
			size = 78;
		end
	end
	
end