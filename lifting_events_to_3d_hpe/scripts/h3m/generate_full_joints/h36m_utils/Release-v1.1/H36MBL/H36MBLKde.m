classdef H36MBLKde < H36MRegressor
% H36MBLKde Implementation of the KDE method of Cortes by a linear 
% approximation based on random fourier features.
%
% method = 'rf'
%        - method for training and testing
% 
% Requires external packages : randfeat(rf), minFunc
properties
    method;
    
    IKType;
    IKParam;
    IKrfopt;
    rfobj_x;
    OKType;
    OKParam;
    OKrfopt;
    rfobj_y;
    
    options;
    gamma;
    verbose;
    
    Kx_inv;
    W;
    W_init;
    
    Napp_x;
    Napp_y;
    Napp_y_augmented;
    
    Input;
    Output;
    
    pres;
    sample_rate;
    lambda;
		
    lro;
    
		normalizeRF;
		
		dim;
		MAX_SAMPLES;
  end
  
  methods
    function obj = H36MBLKde(varargin)
      obj.method  = 'rf';
      obj.IKType  = 'exp_chi2'; 
      obj.IKParam = 1.7;
      obj.OKType  = 'rbf';
      obj.OKParam = 1e-6;
      obj.verbose = 0;
      obj.gamma = 1e-4;
			obj.lambda = .5;
			obj.normalizeRF = true;
			
      % inference parameters
      options = optimset('GradObj','on');
      options = optimset(options,'LargeScale','on');
      options = optimset(options,'DerivativeCheck','off');
      options = optimset(options,'Display','off');
      options = optimset(options,'MaxIter',50); 
      options = optimset(options,'TolFun',1e-8);
      options = optimset(options,'TolX',1e-8);
      options.LS_init = 2;
      options.LS = 0;
      options.Method = 'qnewton';
      obj.IKrfopt.method = 'chebyshev';
      obj.IKrfopt.Nperdim = 5;
      obj.OKrfopt.method = 'sampling';
      obj.options = options;
      
      obj.MAX_SAMPLES = Inf;
      
      switch obj.method           
        case 'rf'
          obj.Napp_x = 15000;
          obj.Napp_y = 4000;
          obj.OKType = 'rbf';
					        
        otherwise
          error('Unknown method! Only rf is recognised!');
      end
      
      obj = obj.fill_in(varargin{:});
      
      if ~isempty(obj.pres) && sum(obj.pres)>obj.MAX_SAMPLES
        obj.sample_rate = ceil(sum(obj.pres)/obj.MAX_SAMPLES);
      else
        obj.sample_rate = 1;
      end
      obj.traintime = 0;
			obj.testtime = 0;
    end
    
    function obj = train(obj)
      if obj.verbose > 0
        disp('Training...');
      end
      
      N = size(obj.Input,1);

      t = tic;
      switch obj.method
        case 'rf'        
          [obj.W obj.W_init] = obj.lro.Regress(obj.gamma,obj.lambda);
          obj.lro.Hessian = [];
					
        otherwise 
          error('Unknown method! Only rf is recognised!');
      end
      
      obj.traintime = obj.traintime + toc(t);
    end
    
    function [Prediction obj] = test(obj, TestInput)
      
      if obj.verbose > 0
        disp('Testing...');
      end

      N = size(TestInput{1},1);
			D = size(obj.W_init,2);
      Prediction = zeros(N,D);

      t = tic;
			
			% prepare the initialization
      if strcmp(obj.method,'rf') 
        if obj.normalizeRF
					TestInputRF = [ones(size(TestInput{1},1),1) rf_featurize(obj.rfobj_x, TestInput{1})./ sqrt(obj.Napp_x)];
				else
					TestInputRF = [ones(size(TestInput{1},1),1) rf_featurize(obj.rfobj_x, TestInput{1})];
				end
        Wx = TestInputRF*obj.W;
        YInit = TestInputRF * obj.W_init;
			end

			% perform inference
      for i = 1 : N
        if obj.verbose && mod(i, 100) == 0
          disp(['Progress ' num2str(i*100/N) '% | Time ' num2str(toc(t))]);
        end

        switch obj.method
          case 'rf'
            [pred, f] = minFunc(@linearFunctionRF, YInit(i,:)', obj.options, obj, Wx(i,:)');
            Prediction(i,:) = pred';
					
          otherwise
            error('Unknown method! Only rf is recognised!');
        end
			end

			obj.testtime = obj.testtime + toc(t);
    end
  
    function obj = update(obj, X, Y)
			t = tic;
			
      switch obj.method 
				case 'rf'
          if isempty(obj.rfobj_x)
            obj.rfobj_x = rf_init(obj.IKType, obj.IKParam, size(X{1},2), obj.Napp_x, obj.IKrfopt);
            obj.rfobj_y = rf_init(obj.OKType, obj.OKParam, size(Y{1},2), obj.Napp_y, obj.OKrfopt);
          end

          Mx = rf_featurize(obj.rfobj_x, X{1});
          My = rf_featurize(obj.rfobj_y, Y{1});
					
					if obj.normalizeRF
						Mx = Mx ./ sqrt(obj.Napp_x);
						My = My ./ sqrt(obj.Napp_y);
					end
					
          if isempty(obj.lro)
            obj.lro = LinearKDERegressor_Data2(Mx,My,Y{1});
          else
            obj.lro = plus(obj.lro,LinearKDERegressor_Data2(Mx,My,Y{1}));
					end
				otherwise
					error('Unknown method!');
			end
			
			obj.traintime = obj.traintime + toc(t);
    end
  end
  
  methods	
    function [f grad] = linearFunctionRF(y, kde, Wx)
      y = y';
			switch kde.OKType
				case 'linear'
					f = norm(y)^2 - 2*y*Wx;
					if nargout == 2
						grad = 2*y'-2*Wx; 
					end
					
				otherwise
					TargetRF = rf_featurize(kde.rfobj_y, y);
					if kde.normalizeRF
						TargetRF = TargetRF ./ sqrt(kde.Napp_y);
					end
					f = 1 - 2*TargetRF*Wx;
					
					if nargout == 2
						if kde.normalizeRF
							grad = 2*(kde.rfobj_y.omega/sqrt(kde.Napp_y) *sqrt(2)* (sin(y*kde.rfobj_y.omega+ kde.rfobj_y.beta'*2*pi)'.* Wx)); 
						else
							grad = 2*(kde.rfobj_y.omega *sqrt(2)* (sin(y*kde.rfobj_y.omega+ kde.rfobj_y.beta'*2*pi)'.* Wx)); 
						end
					end
			end
    end
  end
end