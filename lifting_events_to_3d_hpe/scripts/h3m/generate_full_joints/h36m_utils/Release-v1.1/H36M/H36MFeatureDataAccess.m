classdef H36MFeatureDataAccess < H36MDataAccess
  properties
    FileName;
    VarName;
    Buffer;
		Type;
  end
  
  methods
    function obj = H36MFeatureDataAccess(filename,varname,perm)
      obj.FileName = filename;
      obj.VarName = varname;
      obj.Exists = false;
      obj.Permanent = false;
			
			obj.Type = filename(end-2:end);
			if exist(filename,'file')
				switch obj.Type
					case 'cdf'
            try
              a = cdfread(filename,'Variable', {varname});
            catch e
              error(['Error reading ' filename]);
            end
						obj.Buffer = a{1};
					
					case 'mat'
            try
              load(filename);
            catch e
              error(['Error reading ' filename]);
            end
            
						eval(['obj.Buffer =' varname ';']);
						
					otherwise
						error('Unsupported format!');
				end
				obj.Exists = true;
				obj.Permanent = false;
      else
        obj.Buffer = [];
				if exist('perm','var')
					obj.Permanent = perm;
				end
			end
		end
    
		function flag = exist(obj)
			flag = obj.Exists;
		end
		
    function [F obj] = getFrame(obj, fno)
			switch obj.Type
				case 'cdf'
					F = obj.Buffer(fno,:);
				case 'mat'
		      F = obj.Buffer{fno}; 
			end
    end
    
    function obj = putFrame(obj,fno,F)
			switch obj.Type
				case 'cdf'
					obj.Buffer(fno,:) = F;
				case 'mat'
					obj.Buffer{fno} = F;
			end
    end
       
    function save(obj)
			if obj.Permanent 
				try 
					switch obj.Type
						case 'cdf'
							Feat = obj.Buffer;
							cdfwrite(obj.FileName,{obj.VarName,Feat});
						case 'mat'
							eval([obj.VarName '= obj.Buffer;']);
							save(obj.FileName,'-v7.3',obj.VarName);
					end
				catch E
					error(['Error in file: ' obj.FileName]);
				end
			end
    end
  end
end
