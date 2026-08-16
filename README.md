# Behavioral Reprogramming of Open-Weights Models

[![arXiv](https://img.shields.io/badge/arXiv-2608.13069-b31b1b.svg)](https://arxiv.org/abs/2608.13069)
[![License](https://img.shields.io/badge/Status-Research%20%26%20Commercial%20Asset-blue.svg)]()

This repository contains the source code, Slurm execution scripts, configurations, convergence logs, and aggregated metrics supporting the paper:
**"Behavioral Modification Boundaries of Open-Weight Large Language Models Under Direct Preference Optimization"**.

---

## Repository Structure

* **Python Experiment Runners (`exp1_...` to `exp5_...`, `collect_results.py`):** Core scripts for learning curve sweeps, base vs. instruct benchmarks, cross-lingual transfer, and personality stress testing.
* **HPC Execution Scripts (`.sh`, `submit_all.sh`):** Slurm workload manager configurations (`sbatch`) optimized for distributed HPC clusters (EuroHPC / Leonardo).
* **Logs & Metrics (`.csv`, `.json`, `*.out`):** Raw cluster logs, execution traces, sweep summaries, and aggregated benchmarks across all 6 experimental runs.
* **Visualizations (`.png`):** Learning curves, epoch ablation dynamics, and benchmark comparisons.

---

## Commercial Licensing & Model Weights

> **Notice:** This repository hosts the experimental verification pipeline, logs, and reproduction scripts. The pre-trained/fine-tuned model checkpoints (weights), proprietary alignment datasets, and industrial avatar integration pipelines are maintained as a commercial deep-tech asset.

For enterprise deployment, technology transfer, or full IP acquisition inquiries:
* **Paper / Research:** [arXiv:2608.13069](https://arxiv.org/abs/2608.13069)
* **Inquiries:** Contact via [LinkedIn](https://www.linkedin.com)
