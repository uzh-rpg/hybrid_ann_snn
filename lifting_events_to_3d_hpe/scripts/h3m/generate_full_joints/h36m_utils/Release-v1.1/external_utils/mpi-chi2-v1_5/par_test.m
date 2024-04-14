function par_test()
% need to export before runing matlab (export OMP_NUM_THREADS=8)
data = rand(200, 5000);

%maxNumCompThreads()

t = tic();
d = chi2_mex_serial(single(data),single(data));
time_serial = toc(t)

t2 = tic();
sym = true;

maxNumCompThreads(8);
setenv('OMP_NUM_THREADS', '8');

d2 = chi2_mex(single(data),single(data), sym);
time_parallel = toc(t2)

all(all(d==d2))
