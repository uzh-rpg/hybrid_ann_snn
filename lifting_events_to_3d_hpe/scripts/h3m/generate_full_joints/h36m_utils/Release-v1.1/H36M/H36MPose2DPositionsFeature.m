classdef H36MPose2DPositionsFeature < H36MPoseFeature
	methods
		function obj    = H36MPose2DPositionsFeature(varargin)
			obj.Dimensions= 2;
      obj.Type			= H36MPoseFeature.POSITIONS_TYPE;
			obj.Part			= [];
      obj.Relative	= false;
      obj.Monocular = false;
      obj.Symmetric = false;
      obj.Serialize = true;
			
      obj = obj.fillin(varargin{:});
      
      obj.FeaturePath = '/MyPoseFeatures/';
      
      obj.FeatureName = sprintf('D%d_%s',obj.Dimensions,obj.Type);
      
%       obj.Extension = '.joints.mat';
			obj.Extension = '.cdf';
			
			obj.RequiredFeatures{1} = H36MRawPoseFeature();
		end
		
		function NFeat	= process(obj, Feat, Subject, Camera)
			Feat            = TransformAngles2BVH(Feat(:,4:end),1:size(Feat,1),Feat(:,1:3));
			Feat            = skel2xyz(Subject.AnglesSkel, Feat);
			NFeat           = Camera.project(Feat);
			NFeat           = NFeat(:)';
		end
		
		function Feat		= normalize(obj, Feat, Subject, Camera)
			if obj.Symmetric
				W = Camera.getResolution();
				Feat(:,1:2:end) = W - Feat(:,1:2:end) + 1;
				NFeat = Feat;
				Feat(:,(17-1)*2+1:24*2) = NFeat(:,(25-1)*2+1:32*2);
				Feat(:,(25-1)*2+1:32*2) = NFeat(:,(17-1)*2+1:24*2);
				Feat(:,(2-1)*2+1:6*2)		= NFeat(:,(7-1)*2+1:11*2);
				Feat(:,(7-1)*2+1:11*2)	= NFeat(:,(2-1)*2+1:6*2);
			end
			
			if obj.Relative
				Feat(:,1:end) = Feat(:,1:end) - repmat(Feat(:,1:obj.Dimensions),[1 length(Subject.getPosSkel.tree)]);
			end
			
			if ~isempty(obj.Part)
				[Feat skel] = obj.select(Feat, Subject.get2DPosSkel, obj.Part);
			else
				skel = Subject.get2DPosSkel();
			end
		end
		
		function Pos		= toPositions(obj, Data, skel)
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
          joints = 18:24;% p/p2/a fine
        case 'rightarm'
          joints = 26:32;% p/p2/a fine
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
			
%       reader = H36MFeatureDataAccess([folder Sequence.getName obj.Extension],'Meas',true);
			reader = H36MFeatureDataAccess([folder Sequence.getName obj.Extension],'Pose',obj.Serialize);
		end
		
		function 		size		= getFeatureSize(obj)
			size = 64;
		end
	end
end