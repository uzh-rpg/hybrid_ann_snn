#!/bin/csh

module load apps/matlab
setenv OMP_NUM_THREADS 8
matlab -nodisplay -nodesktop -nosplash -r "parallel_test()"

