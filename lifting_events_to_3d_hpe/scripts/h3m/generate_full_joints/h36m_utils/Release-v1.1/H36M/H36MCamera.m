classdef H36MCamera
  properties
    Number;
    Name;
    Params;
    Subject;
    Resolution;
    
    R; 
    T; 
    f; 
    c; 
    k; 
    p; 
  end
  
  methods
    function obj = H36MCamera(db, s, c)            
      obj.Name = num2str(db.getCameraName(c));
      obj.Number = c;
      obj.Subject = s;
      
      obj.Resolution = db.getResolution(s,c);
      
      if s == 0
        obj.Params = 0.25* (db.loadCamera(1,1)+db.loadCamera(1,2)+db.loadCamera(1,3)+db.loadCamera(1,4));
        obj.R = eye(3);
        obj.T = [0 0 0];
      else
        obj.Params = db.loadCamera(s,c);
        obj.R = rotationMatrix(obj.Params(1),obj.Params(2),obj.Params(3),'xyz')';%[1 0 0; 0 cos(obj.Params(1)) -sin(obj.Params(1)); 0 sin(obj.Params(1)) cos(obj.Params(1))] * [cos(obj.Params(2)) 0 sin(obj.Params(2)); 0 1 0; -sin(obj.Params(2)) 0 cos(obj.Params(2))] * [cos(obj.Params(3)) -sin(obj.Params(3)) 0; sin(obj.Params(3)) cos(obj.Params(3)) 0; 0 0 1];%
        obj.T = obj.Params(4:6);
      end
      obj.f = obj.Params(7:8);
      obj.c = obj.Params(9:10);
      obj.k = obj.Params(11:13);
      obj.p = obj.Params(14:15);
    end
    
    function [W H] = getResolution(obj)
      W = obj.Resolution(1);
      H = obj.Resolution(2);
    end
    
    function [PX D] = project(obj, X)      
      if size(X,1)==1
        X = reshape(X,[3 size(X,2)/3])';
      end
      [PX D] = ProjectPointRadial(X, obj.R, obj.T, obj.f, obj.c, obj.k, obj.p);
      PX = PX';
      PX = PX(:);
    end
  end
end