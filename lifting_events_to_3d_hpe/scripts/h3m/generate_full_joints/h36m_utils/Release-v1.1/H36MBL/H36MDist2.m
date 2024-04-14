function D = H36MDist2(type, X, Y)
% DB4MPDist is the pairwise distance function between elements of X, Y
% X,Y must have same dimensionality
%
% supported are - euclidean and cosine
%								- chi2 courtesy of fuxin
%								- mean 2d and 3d pose
[N dim] = size(X);
[M dim2]= size(Y);

if dim2~=dim
  error('Not same dimensionality!');
end

switch type
  case {'euclidean','cosine'}
    D = pdist2(X,Y,type);
	case 'chi2'
    try 
      D = chi2_mex(X',Y');
    catch E
      disp('Running unoptimized matlab code!');
      [L1, ~] = size(X);
      [L2, ~] = size(Y);
      D = zeros(L1, L2);
      for i = 1: L1
        si = repmat(X(i,:),[L2 1]);
        D(i,:) = sum((si-Y).^2 ./(si+Y +1e-20),2);
      end
    end
  case 'mean-l2-2d'
    D = zeros(N,M);
    for i = 1: dim/2
      D = D + pdist2(X(:,(i-1)*2+1:2*i),Y(:,(i-1)*2+1:2*i),'euclidean');
		end
		D = D/dim*2;
  case 'mean-l2-3d'
    D = zeros(N,M);
    for i = 1: dim/3
      D = D + pdist2(X(:,(i-1)*3+1:3*i),Y(:,(i-1)*3+1:3*i),'euclidean');
		end
		D = D/dim*3;
	case 'max-l2-3d'
		D = zeros(N,M);
    for i = 1: dim/3
      D = max(D,pdist2(X(:,(i-1)*3+1:3*i),Y(:,(i-1)*3+1:3*i),'euclidean'));
		end
  otherwise
    error('unknown type');
end
end