function obj = InitExplicitKernel( kernel, alpha, D, Napp, options )
%INITEXPLICITKERNEL compute kernel based on explicit linear features
%
% kernel - the name of the kernel. Supported options are: 
%             'rbf': Gaussian, 
%             'laplace': Laplacian, 
%             'chi2': Chi-square, 
%             'chi2_skewed': Skewed Chi-square,
%             'intersection', Histogram Intersection, 
%             'intersection_skewed', Skewed Histogram Intersection
% alpha  - the parameter of the kernel, e.g., the gamma in \exp(-gamma ||x-y||) 
%        for the Gaussian.
% D      - the number of dimensions
% Napp 	 - the number of random points you want to sample
% options: options. Including:
%         options.method: 'sampling','signals' or 'nystrom'.
%                         'signals' for [Vedaldi and Zisserman 2012]-type 
%                                   fixed interval sampling. 
%                         'sampling' for [Rahimi and Recht 2007] type of 
%                                    Monte Carlo sampling.
%                         'nystrom' for Nystrom sampling (user has to
%                                   specify the anchors as options.omega). For Nystrom, PCA 
%                                   is only performed if you call rf_pca_featurize.
%                                   If rf_featurize is called, then the program
%                                   simply evaluates the kernel between examples 
%                                   and anchor points.
%         options.Nperdim: Number of samples per dimension for additive
%                          kernels.
%         options.period: The parameter for [Vedaldi and Zisserman
%                         2012]-type fixed interval sampling.
%         options.omega: User-supplied anchors for using the Nystrom method 
%                        (Only useful for the Nystrom method).
%
% copyright (c) 2010-2012
% Fuxin Li - fli@cc.gatech.edu
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

% number of explicit features with which to approximate
if nargin < 4
  Napp = 10; 
end
if ~exist('options','var')
    options = [];
end

switch kernel
    case 'rbf'
        % check
        obj = rf_init('gaussian', alpha, D, Napp, options);
    case 'exp_hel'
        obj = rf_init('exp_hel', alpha, D, Napp, options);
        
    case 'laplace'
        % not verified
        obj = rf_init('laplace', alpha, D, Napp, options);
    
  case 'chi2'
    if ~isfield(options, 'method')
      options.method = 'signals';
      if ~isfield(options,'Nperdim') || isempty(options.Nperdim)
        options.Nperdim = 7;
      end
      if ~isfield(options,'period') || isempty(options.period)
        options.period = 1 / sqrt(2 * (options.Nperdim+1));
      end
    elseif strcmp(options.method,'chebyshev')
        if ~isfield(options,'Nperdim') || isempty(options.Nperdim)
            options.Nperdim = 10;
        end
    end
    obj = rf_init('chi2', alpha, D, Napp, options);
    
  case 'chi2_skewed'
    obj = rf_init('chi2', alpha, D, Napp, options);
    obj.name = 'chi2_skewed';
    
  case 'intersection'
      if ~isfield(options,'Nperdim') || isempty(options.Nperdim)
        options.Nperdim = 7;
      end
      if ~isfield(options,'period') || isempty(options.period)
          options.period = 6e-1;
      end
    obj = rf_init('intersection', alpha, D, Napp, options);
  
  case 'intersection_skewed'
    obj = rf_init('intersection', alpha, D, Napp, options);
    obj.name = 'intersection_skewed';
    % Linear: no approximation, Napp is ignored
    
  case 'linear'
    obj.name = 'linear';
    obj.Napp = D;
    obj.dim = D;
    obj.final_dim = D;
    
  case 'exp_chi2'
    if ~isfield(options,'method')
        options.method = 'chebyshev';
    end
    if ~isfield(options,'Nperdim')
        options.Nperdim = 9;
    end
    if ~isfield(options,'period')
        options.period = 1 / sqrt(2 * (options.Nperdim+1));
    end
    obj = rf_init('exp_chi2', alpha, D, Napp, options);
  otherwise
    error('Unknown kernel');
end

end

