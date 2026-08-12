#!/bin/bash
#SBATCH --job-name=lc_100_32
#SBATCH --account=AIFAC_F02_159
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp1_learning_curves_v3/exp_20260328_175208/logs/lc_f100_r32_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp1_learning_curves_v3/exp_20260328_175208/logs/lc_f100_r32_%j.err

source /leonardo_scratch/large/userexternal/lmalicko/env_lucy/bin/activate
echo "Job: lc_f100_r32 | Node: $SLURM_NODELIST"
python3 /leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp1_learning_curves_v3/exp_20260328_175208/scripts/lc_f100_r32.py
