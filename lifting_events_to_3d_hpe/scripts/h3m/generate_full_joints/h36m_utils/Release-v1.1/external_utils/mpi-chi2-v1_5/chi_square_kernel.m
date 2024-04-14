function K = chi_square_kernel(u, v, sym, gamma)    
    %% apparently can't control n_workers directly, in linux, just a priori from the shell
    %assert(9 > n_workers); 
    %setenv('OMP_NUM_THREADS',int2str(n_workers));
    K = chi2_mex(u',v', sym);
    if(exist('gamma', 'var'))
        K = exp(-gamma.*K);
    end
end
