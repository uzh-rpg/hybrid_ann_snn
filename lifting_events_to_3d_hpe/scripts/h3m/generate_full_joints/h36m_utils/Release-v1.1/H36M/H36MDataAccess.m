classdef H36MDataAccess
	properties
		Exists;
		Permanent;
	end
	
  methods(Abstract)
    F = getFrame(obj, fno);
    putFrame(obj,fno,F);
		flag = exist(obj);
  end
end