classdef H36MRawPoseFeature < H36MFeature
	properties (Constant)
    ANGLES_TYPE = 'Angles';
    POSITIONS_TYPE = 'Positions';
	end
	
	properties
    Dimensions; % 3D/2D
    Type; % Angles / Positions
    Joints; % joint IDs to be kept
    Relative; % relative / absolute position
    Monocular;
    Symmetric;
	end
	
	methods
		function obj    = H36MRawPoseFeature(varargin)
      obj = obj.fillin(varargin{:});	
			obj.FeaturePath = '/MyPoseFeatures/';
			obj.FeatureName = 'RawAngles';
      obj.Extension = '.cdf';
		end
		
    function Pos		= toPositions(obj, Data, skel)
      if strcmp(obj.Type, obj.POSITIONS_TYPE)
        Pos = Data;
      else
        if obj.Dimensions == 2
          Pos = [];
        else
          Pos2 = skel2xyz(skel, Data)';
          Pos  = Pos2(:)';
        end
      end
    end
    
    function Feat    = process(obj, Feat, ~, ~)      
		end
		
		function Feat   = normalize(obj, Feat, ~, ~)
		end
		
		function LimbLengths = computeLimbLengths(obj, Feat, Skel)
			if any(arrayfun(@(x)(~isempty(x.rotInd)),Skel.tree))
				error('Limb lengths only on position skeletons!');
			end
			
			k = 1;
			for i = 1: length(Skel.tree)
				for j = 1: length(Skel.tree(i).children)
					LimbLengths(:,k) = sqrt(sum((Feat(:,Skel.tree(i).posInd) - Feat(:,Skel.tree(Skel.tree(i).children(j)).posInd)).^2,2));
% 					disp([Skel.tree(i).name '---->' Skel.tree(Skel.tree(i).children(j)).name '---' num2str(LimbLengths(1,k))]);
					k = k + 1;
				end
			end
		end
		
		function [v a] = computeJointVelocities(obj, Feat, dt, Skel, LimbLengths)
			% FIXME acceleration doesn't work
			% FIXME a better model would model the kinematic dependencies as well
			if nargin < 4
				LimbLengths = obj.computeLimbLengths(Feat,Skel);
			end
			
			if size(dt,1)==1
				dt = dt * ones(size(Feat,1),1);
			end
			
			v = zeros(size(Feat));
			a = zeros(size(Feat));
			for i = 1: length(Skel.tree)
				for j = 1: length(Skel.tree(i).children)
					parentind = Skel.tree(i).posInd;
					childind = Skel.tree(Skel.tree(i).children(j)).posInd;
					v(2:end,childind) = (Feat(2:end,childind) - Feat(1:end-1,childind))./repmat(dt(2:end),[1 3]);
% 					a(3:end,childind) = (Feat(3:end,childind) - 2*Feat(2:end-1,childind)+Feat(1:end-2,childind))./(repmat(dt(3:end).^2,[1 3]));
					a(3:end,childind) = (Feat(3:end,childind) - 2*Feat(2:end-1,childind)+Feat(1:end-2,childind))./(repmat(dt(3:end).^2,[1 3]));
				end
			end
		end
		
    function [NFeat skel2] = select(obj, Feat, skel, part)
      if strcmp(obj.Type,obj.ANGLES_TYPE)
        type = 'rotInd';
      else
        type = 'posInd';
      end
      
      switch part
        case 'rootpos'
          joints = 1;
          type = 'posInd';
        case 'rootrot'
          joints = 1;
          type = 'rotInd';
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
        case 'lowerbody'
          joints = 1:11;% p/p2/a fine
        case 'body'
          joints = [1 2 7 12 13 14 17 18 25 26];% p/p2/a fine
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
        skel2.tree(i).(type) = p:p+length(skel.tree(joints(i)).(type))-1;
        p = p + length(skel.tree(joints(i)).(type));
        
        % update parents and children
        if strcmp(type,'posInd')
          skel2.tree(i).rotInd = [];
        end
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
      idx = [skel.tree(joints).(type)];
      
      NFeat = Feat(:,idx);
    end
    
    function flag = exist(obj, Sequence)
      flag = exist([Sequence.getPath() filesep obj.FeaturePath Sequence.getBaseName() obj.Extension],'file');
    end
    
    function reader  = serializer(obj, Sequence)
      reader = H36MPoseDataAcess([Sequence.getPath() filesep ...
                obj.FeaturePath obj.FeatureName filesep Sequence.getBaseName obj.Extension]);
		end
		
		function 		size		= getFeatureSize(obj)
			% FIXME 
			size = 0;
		end
	end
end