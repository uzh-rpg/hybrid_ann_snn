classdef H36MRegressor
  % regressor abstract class
  properties
		traintime;
    testtime;
		Ntex; % Number of training examples
	end
	
  methods (Abstract)  
    obj = train(obj);
    obj = test(obj, X, Y);
    obj = update(obj, X, Y);
  end
  
  methods
    function obj = fill_in(obj, varargin)
      for i = 1: 2: length(varargin)
        if isempty(varargin{i})
          continue;
        end
        obj.(varargin{i}) = varargin{i+1};
      end
    end
    
    function save(obj, a)
      fnames = fieldnames(obj);
      for i = 1: length(fnames)
        eval([fnames{i} '=obj.' fnames{i} ';'])
        if i > 1
          save(a,'-append','-v7.3',fnames{i});
        else
          save(a,'-v7.3',fnames{i});
        end
      end
    end
    
    function obj = load(obj,a)
      fnames = fieldnames(obj);
      for i = 1: length(fnames)
        try
          load(a,fnames{i});
          eval(['obj.' fnames{i} '=' fnames{i}  ';'])
        catch e
					warning(['Loading Error. Not loading ' fnames{i} '.']);
        end
      end
    end
  end
end