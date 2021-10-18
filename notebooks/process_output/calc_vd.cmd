#!/bin/bash
#  UGE job for run_sequence.py built Thu Feb 16 09:35:24 PST 2017
#
#  The following items pertain to this script
#  Use current working directory
#$ -cwd
#  input           = /dev/null
#  output          = /u/home/a/ajshajib/Logs/joblog
#$ -o /u/home/a/ajshajib/Logs/joblog.$JOB_ID
#  error           = Merged with joblog
#$ -j y
#  The following items pertain to the user program
#  user program    = /u/home/a/ajshajib/Scripts/run_sequence.py
#  arguments       = mcmc_test
#  program input   = Specified by user program
#  program output  = Specified by user program
#  Parallelism:  1-way parallel
#  Resources requested
#$ -pe dc_* 1
#$ -l h_data=999M,h_rt=24:00:00
#
#$ -M ajshajib@mail
#  Notify at beginning and end of job
#$ -m bea
#  Job is not rerunable
#$ -r n
#  Uncomment the next line to have your environment variables used by SGE
# -V
#
#
echo ""
echo "Job (run_sequence.py) $JOB_ID started on:   "` hostname -s `
echo "Job (run_sequence.py) $JOB_ID started on:   "` date `
echo ""
#
# Run the user program
#
. /u/local/Modules/default/init/modules.sh
module load intel/14.cs
#module load intelmpi/5.0.0
export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=4

module load python/2.7.16
#module load anaconda3/2019.03
module load gcc/7.2.0
#module load intelmpi/5.0.0
#export PYTHONPATH=/u/home/a/ajshajib/python_packages/lib/python2.7/site-packages:/u/home/a/ajshajib/python_packages:
#export LD_LIBRARY_PATH=/u/home/a/ajshajib/mybin/MultiNest/lib:/u/local/compilers/intel-cs/2013.0.028/mpi/intel64/lib:/u/local/compilers/intel-cs/2013.0.028/itac/intel64/itac/slib_impi4:/u/local/compilers/intel-cs/2013.0.028/composer_xe/compiler/lib/intel64:/opt/intel/mic/coi/host-linux-release/lib:/opt/intel/mic/myo/lib:/u/local/compilers/intel-cs/2013.0.028/composer_xe/mpirt/bin/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/ipp/lib/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/mkl/lib/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/tbb/lib/intel64

module list
which mpirun
which python
echo /u/local/compilers/intel-cs/2013.0.028/mpi/intel64/lib:/u/local/compilers/intel-cs/2013.0.028/itac/intel64/itac/slib_impi4:/u/local/compilers/intel-cs/2013.0.028/composer_xe/compiler/lib/intel64:/opt/intel/mic/coi/host-linux-release/lib:/opt/intel/mic/myo/lib:/u/local/compilers/intel-cs/2013.0.028/composer_xe/mpirt/bin/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/ipp/lib/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/mkl/lib/intel64:/u/local/compilers/intel-cs/2013.0.028/composer_xe/tbb/lib/intel64

echo "`which mpirun` -np 1 `which python` /u/home/a/ajshajib/auto_submit/run_sequence.py run_blk_0_0_0_0_0 >& /u/home/a/ajshajib/Logs/output.$JOB_ID"

time `which mpirun` -np 1 `which python`  \
         /u/home/a/ajshajib/process_output/process_output.py 0408_run423_1_2_0_1_0_1_1_0_mod_out.txt composite >& /u/home/a/ajshajib/Logs/output.$JOB_ID


echo ""
echo "job (run_sequence.py) $JOB_ID  finished at:  "Thu Aug 29 19:37:39 PDT 2019
echo ""

