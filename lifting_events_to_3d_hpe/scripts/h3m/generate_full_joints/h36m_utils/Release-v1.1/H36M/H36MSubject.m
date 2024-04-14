classdef H36MSubject
  properties(Hidden)
    Number;
    Name;
    AnglesSkel;
    PositionSkel;
    Position2DSkel;
    Mesh;
    exp_dir;
  end
  
  methods
    function obj = H36MSubject(db, s)      
      obj.Number = s;
      obj.Name = db.getSubjectName(s);
      obj.AnglesSkel = db.getAnglesSkel(s);
      obj.PositionSkel = db.getPosSkel();
      
      obj.exp_dir = db.exp_dir;
      obj.Position2DSkel = obj.get2DPosSkel();
		end
    
		function number = getSubjectNumber(obj)
			number = obj.Number;
		end
		
    function name = getName(obj)
      name = obj.Name;
    end
    
    function skel = getAnglesSkel(obj)
      skel = obj.AnglesSkel;
    end
    
    function skel = getPosSkel(obj)
      skel = obj.PositionSkel;
    end
    
    function [skel obj] = get2DPosSkel(obj)
      if isempty(obj.Position2DSkel)
        skel = obj.PositionSkel;
        for i = 1 : length(skel.tree)
          skel.tree(i).posInd = [(i-1)*2+1 i*2];
        end
        obj.Position2DSkel = skel;
      else
        skel = obj.Position2DSkel;
      end
    end
  
    function s = getScaling(obj)
      scalings = [1 .95 1 .95 1 1.1 .95 1.05 1.05 1.15 1.15];
      s = scalings(obj.Number);
    end
    
    function [mesh obj] = getMesh(obj)
      if isempty(obj.Mesh)
        try
          t = tic;
          fprintf(1, 'Loading mesh...');
          obj.Mesh = loadawobj([obj.exp_dir obj.Name filesep 'Scan' filesep 'Scan.obj']);
          disp('Done.');
        catch e
          disp('Fail.');
        end
        toc(t);
        
        mesh = obj.Mesh;
      end
    end
  end
end