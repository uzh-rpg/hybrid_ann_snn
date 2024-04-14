classdef LinearKDERegressor_Data2 < LinearRegressor_Data2
  properties
    InputTarget2;
		Means;
		Basis;
		Eingenval;
  end
  
  methods
    function obj = LinearKDERegressor_Data2(Input, Target1, Target2, W)
      if ~exist('W','var')
        W = [];
      end
      obj = obj@LinearRegressor_Data2(Input,Target1, W);
      
      [n, d] = size(Input);
      if exist('W','var') && ~isempty(W)
          disp('Using weights');
          % Change to column vector
          if (size(W,2) > size(W,1))
              W = W';
          end
          
          Target2 = bsxfun(@times, Target2, sqrt(W));
      end
      
      obj.InputTarget2 = [sum(Target2); Input'* Target2];
    end
    
    function [Weight1 Weight2] = Regress(obj, Lambda1, Lambda2, Reg_Mat)
      Hes = [obj.N obj.FeatSum';obj.FeatSum obj.Hessian];
      if nargin < 2
        Lambda1 = 1e-8*min(diag(Hes));
        Lambda2 = 1e-8*min(diag(Hes));
      end
      if nargin < 3
        Lambda2 = 1e-8*min(diag(Hes));
      end
      d = size(Hes,1);

%             Weight = cell(size(obj.InputTarget,2),1);
      if exist('Reg_Mat','var') && ~isempty(Reg_Mat)
          Weight1 = (Hes + Lambda1*[1 zeros(1,d-1);zeros(d-1,1) Reg_Mat])\obj.InputTarget;
      else
        Weight1 = (Hes + Lambda1*eye(d))\obj.InputTarget;
        Weight2 = (Hes + Lambda2*eye(d))\obj.InputTarget2;
      end
    end
     
    function obj = plus(obj,B)
			if strcmp(class(B),'LinearKDERegressor_Data2')
	            obj.Hessian = obj.Hessian + B.Hessian;
                obj.FeatSum = obj.FeatSum + B.FeatSum;
                obj.N = obj.N + B.N;
                obj.InputTarget = obj.InputTarget + B.InputTarget;
                obj.InputTarget2 = obj.InputTarget2 + B.InputTarget2;
            else
				error('Only supports adding LinearRegressor_Data class objects.');
			end
    end
    
    function obj = mtimes(A,B)
        if strcmp(class(A), 'LinearKDERegressor_Data2')
            obj = A;
            factor = B;
        else
            obj = B;
            factor = A;
        end
        obj.Hessian = factor * obj.Hessian;
        obj.FeatSum = factor * obj.FeatSum;
        obj.N = factor * obj.N;
        obj.InputTarget = factor * obj.InputTarget;
        obj.InputTarget2 = factor * obj.InputTarget2;
		end
		
		function obj = do_pca(obj, ndims)
			t = tic();
			obj.Means = obj.FeatSum ./ obj.N;
			[Basis, Eigenval] = mex_dsyev(obj.Hessian - obj.N .* (Means * Means'));
			obj.Basis = fliplr(Basis);
			obj.Basis = Basis(:,1:ndims);
			disp('Time for Eigenvectors: ');
			toc(t);
		end
  end
end