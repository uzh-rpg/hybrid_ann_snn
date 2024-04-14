classdef H36MTOFDataAccess < H36MDataAccess
  properties
    Range;
		Intensity;
    Index;
    Indicator;
		Flag;
		Block;
  end
  
  methods
    function obj = H36MTOFDataAccess(Subject, Action, SubAction)
      db = H36MDataBase.instance();
      
			try 
				filename2 = db.getFileName(Subject,Action,SubAction);
				tmp = cdfread([db.exp_dir db.getSubjectName(Subject) filesep 'TOF' filesep filename2 '.cdf'], 'Variable',{'RangeFrames','Indicator','Index','IntensityFrames'});
				obj.Range = tmp{1};
				obj.Intensity = tmp{4};
				obj.Index = tmp{3};
				obj.Indicator = tmp{2};
				obj.Flag = true;
			catch e
				disp('TOF not found!');
				obj.Flag = false;
			end
      
    end
    
    function [Range obj Intensity] = getFrame(obj, frames)
			Range = obj.Range(:,:,obj.Index(frames));
			Intensity = obj.Intensity(:,:,obj.Index(frames));
      Intensity(Intensity>2300) = 2300;
    end
    
		function putFrame(~,~,~)
			% READ ONLY
		end
		
		function flag = exist(obj)
			flag = obj.Flag;
		end
  end
end