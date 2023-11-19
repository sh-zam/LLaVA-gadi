#!/bin/bash
#PBS -q gpursaa
#PBS -j oe
#PBS -l walltime=06:30:00,mem=128GB
#PBS -l wd
#PBS -l ncpus=56
#PBS -l ngpus=4
#PBS -M sharaf.zaman@anu.edu.au
#

module load python3/3.10.0
export PYTHONPATH=`realpath ../.local/lib/`
source venv/bin/activate

/scratch/dg97/sz7583/LLaVA/scripts/v1_5/pretrain.sh
