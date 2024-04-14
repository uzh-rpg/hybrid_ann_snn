function as = compute_aseries_vec(x,len)
% The first one in as is a_0, rest follows
    if ~exist('len','var') || isempty(len)
        len = 50;
    end
    N = size(x,1);
    lgx = log(x);
    lgxpi = lgx .^2 / pi^2;
    as = zeros(N,len+1);
    as(:,1) = 2*sech(lgx./2);
    % seems like it should be like this
    as(:,2) = -lgxpi.*as(:,1);
    for i=2:len
        % when i=2, as(i) = a_1(x), so need this transformation 
        k = i-1;
        denom = (2 * k + 1) * (k+1);
        as(:,i+1) = ((- lgxpi * 2 + 4*k^2).*as(:,i) - (2*k-1)*(k-1) * as(:,i-1)) ./ denom;
    end
end