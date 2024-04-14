function parallel_test()
% testing if we can do matlabpool

data = rand(200, 5000);

%maxNumCompThreads()

t = tic();
d = chi2_mex_serial(single(data),single(data));
time_serial = toc(t)

sym = true;
t2 = tic();
setenv('OMP_NUM_THREADS','8');
d = chi2_mex(single(data),single(data), sym);
time_parallel = toc(t2)
exit()
