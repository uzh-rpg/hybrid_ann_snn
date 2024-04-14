function fval = compute_fintegral(x1,x2,num_terms)
    f1 = compute_aseries(x1,num_terms);
    f2 = compute_aseries(x2,num_terms);
    fval = 1/4  * f1(1) * f2(1) + sum(f1(2:end).* f2(2:end)) / 2;
end