classdef H36MPoseFeature < H36MFeature
	properties (Constant)
    ANGLES_TYPE = 'Angles';
    POSITIONS_TYPE = 'Positions';
	end
	
	properties
		Dimensions;
		Type			;
		Relative	;
		Part			;
		Monocular ;
		Symmetric ;
		Serialize ;
		Norm		  ;
		NormData	;
	end
	
	methods(Abstract)
		[NFeat	skel]	= normalize(obj, Feat, Subject, ~);
		
		Pos = toPositions(obj, Data, skel);
		[NFeat skel2] = select(obj, Feat, skel, part);
		size		= getFeatureSize(obj);
	end
	methods
		function Feat   = unnormalize(obj,Feat)
		end
	end
end