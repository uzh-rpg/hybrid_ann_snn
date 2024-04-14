classdef H36MPoseUniversal3DPositionsFeature < H36MPoseFeature
	% Universal 3D positions are the 3D joint positions with the same
	% skeleton independent of the subject the data actually came from.
	properties
		UniversalAnglesSkel;
		UniversalPosSkel;
	end
	
	methods
		function obj    = H36MPoseUniversal3DPositionsFeature(varargin)
      obj.Dimensions = 3;
      obj.Type			= H36MPoseFeature.POSITIONS_TYPE;
      obj.Relative	= false;
      obj.Part			= [];
      obj.Monocular = false;
      obj.Symmetric = false;
      obj.Serialize = true;
			obj.UniversalAnglesSkel = H36MDataBase.instance().getUniversalAnglesSkel;
			obj.UniversalPosSkel = H36MDataBase.instance().getUniversalPosSkel;
			
      obj = obj.fillin(varargin{:});
      
      obj.FeaturePath = '/MyPoseFeatures/';
      
			if obj.Monocular
				mono = '_mono';
			else
				mono = '';
			end
			
	    obj.FeatureName = sprintf('D%d_%s%s_universal',obj.Dimensions,obj.Type,mono);
			
%       obj.Extension = '.joints.mat';
			obj.Extension = '.cdf';
			
			obj.RequiredFeatures{1} = H36MRawPoseFeature();
		end
		
		function size = getFeatureSize(obj)
			size = 96;
		end
		
		function NFeat		= process(obj, Feat, ~, Camera)
			NFeat           = TransformAngles2BVH(Feat(:,4:end),1:size(Feat,1),Feat(:,1:3));
			NFeat           = skel2xyz(obj.UniversalAnglesSkel, NFeat)';
			NFeat           = NFeat(:)';
			if obj.Monocular
				NFeat         = TransformJointsPositions(NFeat(:,1:end), NFeat(:,1:3), Camera.T, Camera.R);          
			end			
		end
		
		
		function NFeat		= normalize(obj, NFeat, ~, Camera)		
			if obj.Symmetric 
				if obj.Monocular
					Camera0				= H36MCamera(H36MDataBase.instance(), 0,1);
					NFeat					= reshape(NFeat, [3, length(NFeat)/3])';
					[Proj D radial tan r2] = ProjectPointRadial(NFeat, Camera0.R, Camera0.T, Camera.f, Camera.c, Camera.k, Camera.p);
					W							= Camera.getResolution();
					Proj(:,1)			= W-Proj(:,1)+1;
					NFeat					= ProjectPointRadial_inverse(Proj, Camera0.R, Camera0.T, Camera.f, Camera.c, Camera.p, r2, radial, tan, D);
					NFeat					= NFeat';
					NFeat					= NFeat(:)';
				else
					NFeat					= reshape(NFeat, [3, length(NFeat)/3])';
					[Proj D radial tan r2] = ProjectPointRadial(NFeat, Camera.R, Camera.T, Camera.f, Camera.c, Camera.k, Camera.p);
					W							= Camera.getResolution();
					Proj(:,1)			= W-Proj(:,1)+1;
					NFeat					= ProjectPointRadial_inverse(Proj, Camera.R, Camera.T, Camera.f, Camera.c, Camera.p, r2, radial, tan, D);
					NFeat					= NFeat';
					NFeat					= NFeat(:)';
				end
			end
			
			if obj.Relative
				NFeat(:,1:end) = NFeat(:,1:end) - repmat(NFeat(:,1:obj.Dimensions),[1 length(obj.UniversalPosSkel.tree)]);
			end
			
			if ~isempty(obj.Part)
				[NFeat skel] = obj.select(NFeat, obj.UniversalPosSkel, obj.Part);
			else
				skel = obj.UniversalPosSkel;
			end
		end
				
		function Pos		= toPositions(obj, Data, ~)
			Pos = Data;
		end
				
		function [NFeat skel2] = select(obj, Feat, skel, part)
       
			% These are suprious nodes that we short-circuit
			skel.tree(14).children = [skel.tree(14).children 18 26];
			skel.tree(18).parent = 14;
			skel.tree(26).parent = 14;
			skel.tree(1).children = [skel.tree(1).children 13];
		  skel.tree(13).parent = 1;
			
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
          joints = [1 2 3 4 7 8 9 13 14 15 16 18 19 20 26 27 28];% p/p2/a fine
        otherwise
          error('Unknown');
      end
      
      skel2 = skel;
      skel2.tree = skel.tree(1);
      p = 1;
      for i = 1: length(joints)
        % take node corresponding to joint
        skel2.tree(i) = skel.tree(joints(i));
        skel2.tree(i).children = [];
        
        % update the channels
        skel2.tree(i).posInd = p:p+length(skel.tree(joints(i)).posInd)-1;
        p = p + length(skel.tree(joints(i)).posInd);
        
        % update parents and children
        skel2.tree(i).rotInd = [];
				skel2.tree(i).parent = find(skel.tree(joints(i)).parent==joints);
				
        for j = 1: length(skel.tree(joints(i)).children)
          a = find(skel.tree(joints(i)).children(j)==joints);
          if ~isempty(a)
            skel2.tree(i).children = [skel2.tree(i).children a];
          end
        end
        if isempty(skel2.tree(i).parent)
          skel2.tree(i).parent = 0;
        end
      end
      idx = [skel.tree(joints).posInd];
      
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
	end
	
end