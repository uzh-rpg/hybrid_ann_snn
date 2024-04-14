% LinearRegressor_Data class supports an out-of-core linear regressor where the data cannot be loaded into memory.
% Usage example:

% L1 = LinearRegressor_Data(Input, Target, Weights);

%% Input is an n*D data matrix, with n training points in D dimensions. Target is n * c, where c is the number of output dimensions. Weights is an optional n * 1 vector for the weight on each training point.
%
% %Other training data can be added to L1:
% 
% L1 = L1 + LinearRegressor_Data(Input2, Target2, Weights2);
%
% %You may also do a weighted addition since scalar multiplication is also implemented:
%
% L1 = 3 * L1 + 2 * LinearRegressor_Data(Input2, Target2, Weights2);
%
% %After building L1 up, you can choose from different regressors. Now implemented are ridge regression, group lasso and PCA. For ridge regression, just use:
%
% L1.Regress(Lambda);
%% Lambda is the ridge regression weight.
%
%% For PCA, use:
% L1.PCA(d);
%% to get d eigenvectors
%
%% For Group Lasso, use:
% L1.GroupLasso(Lambda, groups);
%% where Lambda is the regularization parameter and groups specify which variates are grouped together. groups = 1:D gives regular LASSO.
% Currently this version only supports dense matrices.

classdef LinearRegressor_Data2 < handle
    properties
        Hessian
        FeatSum
        N
        InputTarget
    end
    methods
        function obj = plus(obj,B)
			if strcmp(class(B),'LinearRegressor_Data2')
	            obj.Hessian = obj.Hessian + B.Hessian;
                obj.FeatSum = obj.FeatSum + B.FeatSum;
                obj.N = obj.N + B.N;
                obj.InputTarget = obj.InputTarget + B.InputTarget;
            else
				error('Only supports adding LinearRegressor_Data class objects.');
			end
        end
        function obj = mtimes(A,B)
            if strcmp(class(A), 'LinearRegressor_Data2')
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
        end
        function obj = times(obj,B)
            obj = mtimes(obj,B);
        end
        function obj = rdivide(obj,B)
            obj = mtimes(obj,1./B);
        end
        function obj = mrdivide(obj,B)
            obj = mtimes(obj,1./B);
        end
        function obj = LinearRegressor_Data2(Input, Target, W)
            [n, d] = size(Input);
            if exist('W','var') && ~isempty(W)
                disp('Using weights');
                % Change to column vector
                if (size(W,2) > size(W,1))
                    W = W';
                end
                Target = bsxfun(@times, Target, sqrt(W));
                Input = repmat(sqrt(W), 1, d) .* Input;
            end
            obj.Hessian = Input'*Input;
            obj.FeatSum = sum(Input)';
            obj.N = n;
            %    HessiHessian = [BiasVec'*BiasVec BiasVec'*Input; Input'*BiasVec Input'*Input];
            obj.InputTarget = [sum(Target); Input'* Target];
        end
        function Weight = Regress(obj, Lambda, Reg_Mat)
            Hes = [obj.N obj.FeatSum';obj.FeatSum obj.Hessian];
            if nargin < 2
                Lambda = 1e-8*min(diag(Hes));
            end
            d = size(Hes,1);
            
%             Weight = cell(size(obj.InputTarget,2),1);
            if exist('Reg_Mat','var') && ~isempty(Reg_Mat)
                Weight = (Hes + Lambda*[1 zeros(1,d-1);zeros(d-1,1) Reg_Mat])\obj.InputTarget;
            else
                Weight = (Hes + Lambda*eye(d))\obj.InputTarget;
            end
        end
        function Weight = GroupLasso(obj, Lambda, groups)
            Hes = [obj.N obj.FeatSum';obj.FeatSum obj.Hessian];
            if nargin < 2
                Lambda = 1e-8*min(diag(Hes));
            end
            d = size(Hes,1);
            
            glOpts.mFlag=1;
            glOpts.lFlag=0;
            glOpts.q=2;
            if size(groups,1) > size(groups,2)
                groups = groups';
            end
            glOpts.ind = [0 1 cumsum(groups)+1];
            Weight = cell(size(obj.InputTarget,2),1);
            for i=1:size(obj.InputTarget,2)
                disp(['Regressing the ' int2str(i) '-th output']);
                t = tic();
                Weight{i} = glLeastR(Hes, obj.InputTarget(:,i), Lambda, glOpts);
                toc(t);
            end
        end
		% Perform PCA on Hessian, Return Mean vector as well
		function [Basis, Means, Eigenval] = PCA(obj, ndims)
		% Eigenvectors, I should probably implement a custom version because MATLAB symmetric eigenvalues don't use MKL and is rather slow.
            t = tic();
			Means = obj.FeatSum ./ obj.N;
			[Basis, Eigenval] = mex_dsyev(obj.Hessian - obj.N .* (Means * Means'));
            Basis = fliplr(Basis);
            Basis = Basis(:,1:ndims);
            disp('Time for Eigenvectors: ');
            toc(t);
		end
    end
end
