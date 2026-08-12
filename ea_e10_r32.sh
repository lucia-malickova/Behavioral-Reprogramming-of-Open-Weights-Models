#!/bin/bash
#SBATCH --job-name=ea_10_32
#SBATCH --account=AIFAC_F02_159
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp5_epoch_ablation_v3/exp_20260328_180256/logs/ea_e10_r32_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp5_epoch_ablation_v3/exp_20260328_180256/logs/ea_e10_r32_%j.err

source /leonardo_scratch/large/userexternal/lmalicko/env_lucy/bin/activate
echo "Job: ea_e10_r32 | Node: $SLURM_NODELIST"
python3 /leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp5_epoch_ablation_v3/exp_20260328_180256/scripts/ea_e10_r32.py
