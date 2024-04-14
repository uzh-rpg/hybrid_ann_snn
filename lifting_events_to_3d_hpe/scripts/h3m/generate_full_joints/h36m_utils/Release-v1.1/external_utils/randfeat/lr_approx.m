function [ P Q R] = lr_approx( M, scheme )
%LR_APPROX compute low rank approximation of the 

switch scheme
  case 'eigdec'
    [ Q P R] = svd(M);
    R = R';
    
  otherwise
    error('not supported');
end

end

