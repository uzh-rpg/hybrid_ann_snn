function F = rf_featurize(obj, X, Napp)
%RF_FEATURIZE returns the features corresponding to the inputs X
%
% obj   - random feature object initialized by rf_init.
% X     - n * d input matrix. n is the number of examples and d is the
% number of features.
% Napp  - specifies the number of features to be extracted. If obj.method is
%       signals then it is specified per dimension as 2*floor(Napp/2)+1 
%       otherwise it is the number of terms in the MC approximation.
%
% copyright (c) 2010-2012
% Fuxin Li - fli@cc.gatech.edu
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de
if ~exist('Napp','var')
    Napp = obj.Napp;
end
% Use mex_featurize if it's exp-chi2 and not Nystrom method.
% if strcmp(obj.name,'exp_chi2') && ~strcmp(obj.method,'nystrom')
%     if isfield(obj,'omega2')
%         obj.omega2 = obj.omega2';
%     end
%     F = mex_featurize(obj, double(X), Napp)';
% else
    [N D] = size(X);
    
    if isfield(obj,'omega') && Napp > size(obj.omega,2) && strcmp(obj.method,'sampling')
        disp(['Warning: selected number of random features ' num2str(Napp) 'more than built-in number of random features ' num2str(size(obj.omega,2)) '.']);
        disp(['Changing the number of random features to ' num2str(size(obj.omega,2)) '.']);
        disp('You can increase the built-in number in rf_init()');
        Napp = size(obj.omega,2);
    end
    
    if D ~= obj.dim
        error('Dimension mismatch!');
    end
    
    switch obj.name
        case {'gaussian','rbf'}
            F = sqrt(2) * (cos( X * obj.omega(:,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi));
        case 'exp_hel'
            if ~isempty(find(X<0,1,'first'))
                error('Error: Input matrices have negative entries in the Hellinger kernel.');
            end
            
            if ~isfield(obj, 'gamma') || obj.gamma == inf
                F = sqrt(2) * (cos( sqrt(X) * obj.omega(:,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi));
            else
                % Hack exp_hel directly because it's so easy to recover it.
                XX = zeros(N,1);
                XX2 = zeros(1,Napp);
                for i=1:N
                    XX(i) = sqrt(X(i,:)) * sqrt(X(i,:))';
                end
                for i=1:Napp
                    XX2(i) = obj.omega(:,i)' * obj.omega(:,i);
                end
                XX = XX - mean(XX);
                XX2 = XX2 - D/2 * log(obj.kernel_param/pi);
                XX2 = XX2 - mean(XX2);
                weights = exp(-XX ./ 4*(obj.gamma + obj.kernel_param));
                weights2 = exp(XX2 * obj.kernel_param * obj.kernel_param / (obj.gamma + obj.kernel_param));
                F = cos(obj.gamma ./ (obj.kernel_param + obj.gamma) .* sqrt(X) * obj.omega(:,1:Napp));
                % Singleton expansion is automatic
                %            F = bsxfun(@times,F,weights);
                % Remove weights2 for now it's too small
                %            F = bsxfun(@times,F,weights2);
            end
        case {'chi2','exp_chi2'}
            if ~isempty(find(X<0,1,'first'))
                error('Error: Input matrices have negative entries in the Chi-square kernel.');
            end
            % only this fourier analytic treatment no mc estimation yet for chi2
            if strcmp(obj.method, 'signals')
                F = zeros(N, D * obj.Nperdim);
                even_odd = zeros(N,obj.Nperdim-1);
                even_odd(:,2:2:end) = - pi / 2;
                for i = 1: D
                    F(:,((i-1)*obj.Nperdim+1):(i*obj.Nperdim)) = sqrt(obj.period) * [sech(0)*sqrt(X(:,i)), ...
                        cos(log(X(:,i))*obj.omega(i,1:(obj.Nperdim-1))+even_odd) .* sqrt(2 * X(:,i) * sech(pi * obj.omega(i,1:(obj.Nperdim-1))))];
                end
                F(isinf(F)) = 0;
                F(isnan(F)) = 0;
            elseif strcmp(obj.method, 'chebyshev')
                try
                    var = load('V.mat');
                    if obj.Nperdim > size(var.V,2)
                        disp(['Nperdim larger than pre-stored PCA coefficients. Reducing Nperdim to ' num2str(size(V,2)) '.']);
                        obj.Nperdim = size(var.V,2);
                    end
                catch
                    var = [];
                end
                if issparse(X)
                    F = spalloc(N, D*obj.Nperdim,nnz(X)*obj.Nperdim);
                else
                    F = zeros(N, D*obj.Nperdim);
                end
                % compute the coefficients
                % cseries type
                for i = 1:D
                    % for each dimension
                    % Small anyway, sparsify won't hurt
                    if ~isempty(var)
                        ck = compute_cseries(X(:,i),200);
                        if issparse(X)
                            ck = ck * sparse(var.V(:,1:obj.Nperdim));
                        else
                            ck = ck * var.V(:,1:obj.Nperdim);
                        end
                    else
                        ck = compute_cseries(X(:,i),obj.Nperdim);
                    end
                    F(:,i:D:end) = ck;
                end
                F(isnan(F)) = 0;
                F(isinf(F)) = 0;
            elseif strcmp(obj.method, 'nystrom')
                if strcmp(obj.name,'exp_chi2')
                    F = EvalKernel(X, obj.omega(:,1:Napp)', obj.name, obj.kernel_param);
                else
                    F = EvalKernel(X, obj.omega(:,1:Napp)', obj.name);
                end
                % Early return with Nystrom. Don't do the next step!
                return;
            end
            F(isinf(F)) = 0;
            F(isnan(F)) = 0;
            %         SF = sqrt(1 - sum(F.^2,2));
            %         SF(SF<0) = 0;
            %         SF(SF==1) = 0;
            if strcmp(obj.name,'exp_chi2')
                %            F = sqrt(2) * (cos( [F SF] * obj.omega2(1:D*obj.Nperdim+1,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi));
                F = sqrt(2) * (cos( F * obj.omega2(1:D*obj.Nperdim,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi));
            end
        case 'chi2_skewed'
            if ~isempty(find(X<0,1,'first'))
                error('Error: Input matrices have negative entries in the Skewed Chi-Square kernel.');
            end
            % skewed multiplicative chi-square kernel
            F = sqrt(2) * cos( log(X+obj.kernel_param) * 0.5 * obj.omega(:,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi);
        case 'intersection_skewed'
            if ~isempty(find(X<0,1,'first'))
                error('Error: Input matrices have negative entries in the Skewed Chi-Square kernel.');
            end
            F = sqrt(2) * cos( log(X+obj.kernel_param) * obj.omega(:,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi);
        case 'laplace'
            F = sqrt(2) * (cos( X * obj.omega(:,1:Napp) + obj.beta(1:Napp,ones(1,N))'*2*pi));
            % Linear is just replicate
        case 'linear'
            F = X;
            
        case 'intersection'
            if ~isempty(find(X<0,1,'first'))
                error('Error: Input matrices have negative entries in the Skewed Chi-Square kernel.');
            end
            F = [];
            for i = 1: D
                cterm = cos(log(X(:,i))*obj.omega(i,:)) .* sqrt(2/pi* X(:,i) * (1./(1 + 4 * obj.omega(i,:).^2)));
                %       cterm(isnan(cterm)) = 0;
                sterm = sin(log(X(:,i))*obj.omega(i,:)) .* sqrt(2/pi* X(:,i) * (1./(1 + 4 * obj.omega(i,:).^2)));
                %       sterm(isnan(sterm)) = 0;
                
                F = [F sqrt(obj.period) * [sqrt(2/pi)*sqrt(X(:,i)), cterm, sterm ]];
            end
            % this is a clean up of the rf. If X is 0 then log(X) becomes infinity
            % and cos(log(X)) is NaN. We correct this in the end by putting 0
            F(isnan(F)) = 0;
            %        F = sqrt(2) * (cos( F * obj.omega2' + repmat(obj.beta'*2*pi,N,1)));
        otherwise
            error('Unknown kernel approximation scheme');
    end
    %    end
end
