# Behavioral Reprogramming of Open-Weights Models

This repository contains the source code, experimental scripts, configurations, and aggregated results supporting the study on behavioral fine-tuning, epoch ablation, and cross-lingual persona transfer.

## Repository Structure

- **Python Scripts (`exp1_...` to `exp5_...`, `collect_results.py`):** Core scripts used for executing individual experiments, handling learning curves, base vs. instruct comparisons, cross-lingual transfer, and personality stress testing.
- **Execution & Batch Scripts (`.sh`, `submit_all.sh`):** Cluster submission and job management scripts configured for high-performance computing (HPC) environments using Slurm (`sbatch`).
- **Data & Logs (`.csv`, `.json`, `.out`):** Aggregated metrics, sweep summaries, convergence logs, and model output evaluations.
- **Visualizations (`.png`):** Generated learning curves, epoch ablation plots, base vs. instruct performance graphs, and inference comparisons.

## Reproduction
To replicate the experimental pipeline or inspect specific sweeps, refer to the individual Python experiment runners and corresponding `.sh` execution scripts.
