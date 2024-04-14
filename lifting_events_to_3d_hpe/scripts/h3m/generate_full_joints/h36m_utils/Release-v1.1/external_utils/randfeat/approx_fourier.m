function fval = approx_fourier(x,z, num_terms)
    as = compute_aseries(x,num_terms);
    z2 = cos([1:num_terms]*z);
    fval = as(1)/2 + z2 * as(2:end);
end