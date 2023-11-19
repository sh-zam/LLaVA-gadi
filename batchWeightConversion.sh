#!/bin/bash
#PBS -q gpursaa
#PBS -j oe
#PBS -l walltime=00:30:00,mem=276GB
#PBS -l wd
#PBS -l ncpus=56
#PBS -l ngpus=4
#

module load python3/3.10.0
export PYTHONPATH=`realpath ../.local/lib/`
source venv/bin/activate

