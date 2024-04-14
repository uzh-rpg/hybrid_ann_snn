function [Proj D radial tan r2] = ProjectPointRadial(P, R, T, f, c, k, p)
    N = size(P,1);
    X = R*(P'-T'*ones(1,N));
    XX = X(1:2,:)./([1; 1]*X(3,:));
    r2 = XX(1,:).^2 + XX(2,:).^2;
    radial = 1 + dot(repmat(k',[1 N]), [r2; r2.^2; r2.^3], 1);
    tan = p(1)*XX(2,:) + p(2)*XX(1,:);
    XXX = XX.*repmat(radial+tan,[2 1]) + [p(2) p(1)]'*r2;
    Proj = ones(N,1)*f .* XXX' + ones(N,1)*c;
    D = X(3,:);
end