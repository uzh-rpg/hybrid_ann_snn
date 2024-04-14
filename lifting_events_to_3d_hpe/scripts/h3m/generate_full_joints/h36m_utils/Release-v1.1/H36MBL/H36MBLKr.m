classdef H36MBLKr < H36MRegressor
  properties
    Input;
    Target;
    lambda;
    alpha;
    IKType;
    IKParam;
    Ntrain; % this is the number of training examples to subsample
    sample_rate;
    dim;
		MAX_SAMPLES;
  end
  
  methods
    function obj = H36MBLKr(varargin)
      obj.lambda = 1e-4;
			
      % chi2 params
      obj.IKType  = 'exp_chi2';
      obj.IKParam = 1.7;

      obj.MAX_SAMPLES = 40000;
      
      obj = obj.fill_in(varargin{:});
      
      if obj.Ntex > obj.MAX_SAMPLES
        obj.sample_rate = ceil(obj.Ntex/obj.MAX_SAMPLES);
      else
        obj.sample_rate = 1;
			end
			obj.traintime = 0;
			obj.testtime = 0;
    end
    
    function [obj Pred] = train(obj)
      t = tic; 
			
			K         = EvalKernel(obj.Input,obj.Input,obj.IKType,obj.IKParam);

			obj.alpha = (K + obj.lambda*eye(size(K,1)))\obj.Target;
			
			if nargout == 2
				Pred      = K*obj.alpha;
			end
			
      obj.traintime = obj.traintime+toc(t);
    end
    
    function [Pred obj] = test(obj, X)
			t = tic; 
			K    = EvalKernel(X{1},obj.Input,obj.IKType,single(obj.IKParam));
      Pred = K*obj.alpha;
			obj.testtime = obj.testtime+toc(t);
    end
    
    function obj = update(obj, X, Y)
      if size(X,1) ~= size(Y,1)
        error('Training set doesn''t match!');
			end
      
			obj.Input  = [obj.Input;  X{1}(1:obj.sample_rate:end,:)];
			
			
			obj.Target = [obj.Target; Y{1}(1:obj.sample_rate:end,:)];
			
    end
  end
end
