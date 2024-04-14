classdef H36MVideoDataAccess < H36MDataAccess
	% H36MVideoDataAccess - class for loading video data
	% two engines are available one using the videoutils toolbox and another
	% using matlab's videoreader.
	
  properties
    Reader;
		Type;
  end
  methods
    function obj = H36MVideoDataAccess(filename)
			
			if isunix
				obj.Type = 'default';
			elseif ispc
				obj.Type = 'VideoUtils';
			end
			
			switch obj.Type
				case 'VideoUtils'
					obj.Reader = VideoPlayer(filename,'StepInFrames',1);

				case 'default'
					obj.Reader = VideoReader(filename);
					
				otherwise
					error('Unknown!');
			end

		end
    
		function flag = exist(obj)
			switch obj.Type
				case 'default'
					flag = obj.Reader.NumberOfFrames>1;
				case 'VideoUtils'
					flag = obj.Reader.NumFrames>1;
			end
		end
		
    function [F obj] = getFrame(obj, fno)
			switch obj.Type
				case 'default'
					F = obj.Reader.read(fno);
				case 'VideoUtils'
					F = obj.Reader.getFrameAtNum(fno-1);
			end
    end
    
    function putFrame(obj,fno,F)
      error('Not implemented for now!');
		end
		
		function save(obj)
		end
		
		function delete(obj)
			obj.Reader.delete();
		end
  end
end