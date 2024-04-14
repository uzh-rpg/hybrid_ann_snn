# demands a node with all the processors free
qsub -l num_proc=8 -cwd  parallel_test.sh
