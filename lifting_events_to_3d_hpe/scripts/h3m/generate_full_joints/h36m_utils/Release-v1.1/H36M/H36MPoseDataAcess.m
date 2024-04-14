classdef H36MPoseDataAcess < H36MDataAccess
  properties
    FileName;
    Block;
  end
  
  methods
    function obj = H36MPoseDataAcess(filename)
      obj.FileName = filename;      
      obj.Block = [];
			obj.Exists = false;
			try 
% 				obj.Block = load(obj.FileName);
				a = cdfread(obj.FileName,'Variable',{'Pose'});
				obj.Block = a{1};
				obj.Exists = true;
			catch ERR
				error('File Not Found!');
				ERR
			end
    end
    
    function [F obj] = getFrame(obj, fno)
      if isempty(obj.Block)
        try 
          obj.Block = load(obj.FileName);
        catch ERR
          error('File Not Found!');
          ERR
        end
			end
      
			F = obj.Block((fno-1)*4+1,:);
%       F = [obj.Block.World_Root.Position((fno-1)*4+1,[1 2 3]) obj.Block.World_Root.Rotation((fno-1)*4+1,:) obj.Block.Joints((fno-1)*4+1,:)];
    end
    
    function obj = putFrame(obj,fno,F)
      error('Not implemented');
		end
		
		function flag = exist(obj)
			flag = obj.Exists;
		end
		
		function save(obj)
    end
	end
end
