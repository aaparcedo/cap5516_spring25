#!/bin/bash
#SBATCH --nodes=1              # Number of nodes to request
#SBATCH --ntasks=4             # Adjust based on desired worker count
#SBATCH --ntasks-per-node=4    # Match with --ntasks
#SBATCH --cpus-per-task=4      # CPUs per worker
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --time=24:00:00        # Total time to request
#SBATCH --job-name=cap5516     # Job name of SLURM job
#SBATCH --constraint=h100      # Type of GPU (ARCC only)
#SBATCH --nodelist=evc43

# TODO: fill with you logs folder path
export log_file="path/to/cap5516_spring25/segmentation_tumor_mri/logs/${SLURM_JOB_ID}.log"

exec &> $log_file

echo "Starting time: $(date)"

rm "./slurm-${SLURM_JOB_ID}.out"

# TODO: fill with your conda environment bin path
export PATH="/path/to/.conda/envs/cap5516/bin:$PATH"

# TODO: fill with your conda environment python path
path/to/.conda/envs/cap5516/bin/python  main.py

echo "Ending time: $(date)"
