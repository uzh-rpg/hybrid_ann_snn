function [ P ] = ProjectPointRadial_inverse(Proj, R, T, f, c, p, r2, radial, tan, D)
    N = size(Proj,1);
    XXX = (Proj - ones(N,1)*c) ./ (ones(N,1)*f);
    XX = (XXX' - [p(2) p(1)]'*r2) ./ repmat(radial+tan,[2 1]);
    X =  [XX.*([1; 1]*D); D];
    P = (R'*X + T'*ones(1,N))';
end

