function bs = compute_bseries_vec(x, len)
% The first one in bs is b_0, rest follows
    if ~exist('len','var') || isempty(len)
        len = 50;
    end
    lgx = log(x);
    % log x / 4pi
    lgx2 =  lgx ./ (4 * pi); 
    % log^2 x / pi^2
    lgxpi = lgx .^2 / pi^2;
    ns = 0:100; % the series is set to start from 1 in the paper but i that seems wrong from experiments
    N = size(x,1);
    bn = zeros(N,length(ns));
    for i = 1: N
      bn(i,:) = (ns+1/2)./(((ns+1/4).^2 + lgx2(i).^2).*((ns+3/4).^2 + lgx2(i).^2));
      % seems like this would be better
%       bn(i,:) = (ns+1/2)./(((ns+1/4).^2 + 4*lgx2(i).^2).*((ns+3/4).^2 + 4*lgx2(i).^2)); 
    end
    
    bs = zeros(N,len+1);
    bs(:,1) = -sum(bn,2) .* lgx2 * 2/ pi;
%     bs(:,1) = -sum(bn,2) .* lgx2 ./ (2*pi); % seems like it should be like this
    bs(:,2) = -lgxpi.*bs(:,1) - 4 * lgx / pi^2;
    
    for i=2:len
        % when i=2, bs(i) = b_1(x), so need this transformation
        k = i-1;
        denom = (2 * k + 1) * (k+1);
        bs(:,i+1) = (2 * lgxpi .* (-bs(:,i)-4./lgx * (-1)^(k)) - (2*k-1)*(k-1) * bs(:,i-1) ...
                  + 4*k^2 * bs(:,i)) / denom;
    end
end