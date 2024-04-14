classdef H36MBLLinKrr < H36MRegressor
	% H36MBLLinKrr - implementation of kernel ridge regression using random
	% fourier features.
	% 
	% FIXME multiple kernels
  properties
    lambda;
    alpha;
    
    IKType;
    IKParam;
    
    Napp;
    rfobj;
    lrobj;
    
    weights;
    
    basis;
    means;
    eval;
    PCADIM;
    dim;
		options;
		MAX_SAMPLES;
		sample_rate;
		normalizeRF;
  end
  
  methods
    function obj = H36MBLLinKrr(varargin)
			obj.lambda = 10; 
      
      obj.IKType  = 'exp_chi2';
      obj.IKParam = 1.7;
			obj.normalizeRF = true;
			
      options.Nperdim = 10;
      options.method = 'chebyshev';
      
			obj.options = options;

			obj.Napp    = 15000;
			
      obj = obj.fill_in(varargin{:});

			
			if ~isempty(obj.MAX_SAMPLES)&&obj.MAX_SAMPLES<obj.Ntex
        obj.sample_rate = ceil(obj.Ntex/obj.MAX_SAMPLES);
      else
        obj.sample_rate = 1;
			end
			
			obj.rfobj   = rf_init(obj.IKType, obj.IKParam, obj.dim, obj.Napp, obj.options);
			
			obj.traintime = 0;
			obj.testtime = 0;
    end
    
    % compute the weights
    function obj = train(obj)		
      t = tic;
      if obj.lambda == Inf
        obj.weights = Regress(obj.lrobj);  
      else
        obj.weights = Regress(obj.lrobj,obj.lambda);  
      end
      obj.traintime = obj.traintime + toc(t);
      
      obj.lrobj.Hessian = [];
    end
    
    % test
    function [Y obj] = test(obj, X, ~)
			t = tic;
			Xrf = rf_featurize(obj.rfobj, X{1});
			
			if obj.normalizeRF
				Xrf = Xrf ./ sqrt(obj.Napp);
			end
			
      Y = [ones(size(X{1},1), 1) Xrf] * obj.weights;
			obj.testtime = obj.testtime + toc(t);
    end
    
    % update the hessian and the target data
    function obj = update(obj, X, Y)
			t = tic;

			if obj.sample_rate~=1
				Xrf = rf_featurize(obj.rfobj, X{1}(1:obj.sample_rate:end,:));
				Y{1} = Y{1}(1:obj.sample_rate:end,:);
			else
				Xrf = rf_featurize(obj.rfobj, X{1});
			end
				
			if obj.normalizeRF
				Xrf = Xrf ./ sqrt(obj.Napp);
			end
			
			if isempty(obj.lrobj)
				obj.lrobj = LinearRegressor_Data2(Xrf, Y{1});
				obj.traintime = toc(t);
			else
				obj.lrobj = plus(obj.lrobj,LinearRegressor_Data2(Xrf, Y{1}));
				obj.traintime = obj.traintime + toc(t);
			end
    end
  end
end