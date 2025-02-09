#!/bin/bash
#SBATCH --nodes=1              # Number of nodes to request
#SBATCH --ntasks=8             # Adjust based on desired worker count
#SBATCH --ntasks-per-node=8    # Match with --ntasks
#SBATCH --cpus-per-task=4      # CPUs per worker
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --time=02:00:00        # Total time to request
#SBATCH --job-name=cap5516      # Job name of SLURM job
#SBATCH --constraint=v100      # Type of GPU (ARCC only)

export log_file="/home/cap5516.student2/cap5516_spring25/logs/${SLURM_JOB_ID}.log"

exec &> $log_file

echo "Starting time: $(date)"

rm "./slurm-${SLURM_JOB_ID}.out"

export PATH="/home/cap5516.student2/.conda/envs/cap5516/bin:$PATH"

/home/cap5516.student2/.conda/envs/cap5516/bin/python  main.py

echo "Ending time: $(date)"