classdef H36MSequence
  properties
    Subject;
    Action;
    SubAction;
    Camera;
    
    NumFrames;
    
    IdxFrames;
    
    Path;
    Name;
    BaseName;
    
    Feats;
    OutputFeatureAccess;
  end
  
  methods
    function obj = H36MSequence(s, a, sa, c, idx)
      db = H36MDataBase.instance();
      
      obj.Subject = s;
      obj.Action = a;
      obj.SubAction = sa;
      obj.Camera = c;
            
      obj.NumFrames = db.getNumFrames(s,a,sa);
      
      if ~exist('idx','var') || isempty(idx)
        obj.IdxFrames = 1:obj.NumFrames;
      else
        obj.IdxFrames = idx;
        obj.NumFrames = length(idx);
      end
      obj.Path = [db.exp_dir db.getSubjectName(s)];
      obj.Name = db.getFileName(s,a,sa,c);
      obj.BaseName = db.getFileName(s,a,sa);
    end
    
    function idx = getIdxFrames(obj)
      idx = obj.IdxFrames;
    end
    
    function path = getPath(obj)
      path = obj.Path;
    end
    
    function name = getName(obj)
      name = obj.Name;
    end
    
    function name = getBaseName(obj)
      name = obj.BaseName;
    end
    
    function N = getNumFrames(obj)
      N = obj.NumFrames;
    end
    
    function subject = getSubject(obj)
      db = H36MDataBase.instance();
			if obj.Subject>0
				subject = db.getSubject(obj.Subject);
			else
				subject = db.getUniversalSubject;%db.getSubject(db.renders.subjects(obj.Action));
			end
    end
    
    function cam = getCamera(obj, c)
      db = H36MDataBase.instance();
      if ~exist('c','var')
        c = obj.Camera;
      end
      
      [cam db] = db.getCamera(obj.Subject,c);
		end
    
		% subsample is the rate relative to the current sampling
		function obj = subsample(obj, sr)
			obj.IdxFrames = obj.IdxFrames(1:sr:length(obj.IdxFrames));
			obj.NumFrames = length(obj.IdxFrames);
		end
  end
end