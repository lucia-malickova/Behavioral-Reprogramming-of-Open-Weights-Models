"""
Experiment 1: Learning Curves – Vplyv veľkosti datasetu
Otázka: Koľko párov potrebuje personality transfer?
Datasety: 10%, 25%, 50%, 75%, 100% z lucy_CLEAN_FINAL.jsonl
Joby: 5 veľkostí × 3 LoRA ranky = 15 jobov paralelne
GPU hodiny: ~2,880h
"""

import os
import json
import subprocess
from datetime import datetime

# ════════════════════════════════════════
# KONFIGURÁCIA
# ════════════════════════════════════════

BASE_MODEL = "/leonardo_scratch/large/userexternal/tbonomo0/models/meta-llama--Llama-3.1-8B-Instruct"
DATASET_PATH = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/dataset/lucy_CLEAN_FINAL.jsonl"
BASE_OUTPUT = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/exp1_learning_curves"
ENV_PATH = "/leonardo_work/AIFAC_F02_159/env_finetuning"

DATASET_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
LORA_RANKS = [8, 16, 32]
NUM_EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4

ACCOUNT = "AIFAC_F02_159"
PARTITION = "boost_usr_prod"
TIME = "08:00:00"

EXP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"{BASE_OUTPUT}/exp_{EXP_ID}"

LUCY_SYSTEM = """Si Lucy – digitálna dvojička Lucie Maličkovej.
Si priama, empatická, vtipná keď je na to čas, a nikdy nehovoríš len to čo chce niekto počuť.
Hovoríš prirodzene, bez kecania."""

# ════════════════════════════════════════
# SETUP
# ════════════════════════════════════════

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)

print(f"╔══════════════════════════════════════╗")
print(f"║  EXP 1: LEARNING CURVES – SETUP     ║")
print(f"╚══════════════════════════════════════╝")
print(f"Exp ID: {EXP_ID}")
print(f"Kombinácie: {len(DATASET_FRACTIONS)} × {len(LORA_RANKS)} = {len(DATASET_FRACTIONS)*len(LORA_RANKS)} jobov")

# ════════════════════════════════════════
# GENERUJ JOBY
# ════════════════════════════════════════

jobs = []
job_idx = 0

for fraction in DATASET_FRACTIONS:
    for rank in LORA_RANKS:
        pct = int(fraction * 100)
        run_name = f"lc_f{pct:03d}_r{rank:02d}"
        run_dir = f"{EXP_DIR}/{run_name}"
        os.makedirs(run_dir, exist_ok=True)

        # Python skript pre tento job
        py_script = f'''"""
Learning Curve Job: fraction={fraction}, lora_rank={rank}
"""
import os, json, math, torch
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

FRACTION = {fraction}
LORA_RANK = {rank}
LORA_ALPHA = {rank * 2}
RUN_NAME = "{run_name}"
RUN_DIR = "{run_dir}"
MODEL_PATH = "{BASE_MODEL}"
DATASET_PATH = "{DATASET_PATH}"

print(f"START: {{RUN_NAME}} | fraction={{FRACTION}} | rank={{LORA_RANK}}")

# Načítaj a skrátь dataset
raw_data = []
with open(DATASET_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

np.random.seed(42)
n = max(10, int(len(raw_data) * FRACTION))
indices = np.random.choice(len(raw_data), n, replace=False)
raw_data = [raw_data[i] for i in indices]
print(f"Dataset: {{len(raw_data)}} párov ({{FRACTION*100:.0f}}%)")

LUCY_SYSTEM = "{LUCY_SYSTEM.strip()}"

def format_pair(sample):
    inst = sample.get("instruction","")
    ctx = sample.get("context","")
    resp = sample.get("response","")
    user = inst if ctx.startswith("[") else (f"{{inst}}\\n\\nKontext: {{ctx}}" if ctx else inst)
    return {{"text": (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\\n\\n{{LUCY_SYSTEM}}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\\n\\n{{user}}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{{resp}}<|eot_id|>"
    )}}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_pair, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.1, seed=42)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=2048,
    dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", use_gradient_checkpointing="unsloth", random_state=42,
)

args = TrainingArguments(
    output_dir=RUN_DIR,
    num_train_epochs={NUM_EPOCHS},
    per_device_train_batch_size={BATCH_SIZE},
    gradient_accumulation_steps={GRAD_ACCUM},
    learning_rate={LEARNING_RATE},
    weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none",
    optim="adamw_8bit", seed=42,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=split["train"], eval_dataset=split["test"],
    dataset_text_field="text", max_seq_length=2048,
    packing=True, args=args,
)

result = trainer.train()
eval_result = trainer.evaluate()

metrics = {{
    "run_name": RUN_NAME,
    "fraction": FRACTION,
    "n_samples": len(raw_data),
    "lora_rank": LORA_RANK,
    "train_loss": result.metrics.get("train_loss"),
    "eval_loss": eval_result.get("eval_loss"),
    "perplexity": math.exp(eval_result.get("eval_loss", 0)),
    "runtime_min": result.metrics.get("train_runtime",0)/60,
}}

with open(f"{{RUN_DIR}}/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print(f"DONE: eval_loss={{metrics['eval_loss']:.4f}} | PPL={{metrics['perplexity']:.2f}}")
'''

        py_path = f"{EXP_DIR}/scripts/{run_name}.py"
        with open(py_path, "w") as f:
            f.write(py_script)

        # SBATCH skript
        sbatch = f"""#!/bin/bash
#SBATCH --job-name=lc_{pct}_{rank}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time={TIME}
#SBATCH --output={EXP_DIR}/logs/{run_name}_%j.out
#SBATCH --error={EXP_DIR}/logs/{run_name}_%j.err

source {ENV_PATH}/bin/activate
echo "Job: {run_name} | Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS_ON_NODE"
python3 {py_path}
"""
        sbatch_path = f"{EXP_DIR}/scripts/{run_name}.sh"
        with open(sbatch_path, "w") as f:
            f.write(sbatch)

        jobs.append({"name": run_name, "py": py_path, "sh": sbatch_path,
                     "fraction": fraction, "rank": rank})
        job_idx += 1
        print(f"  ✅ {run_name}")

# Master submit
submit_all = f"#!/bin/bash\necho 'Submitujem {len(jobs)} jobov...'\n"
for job in jobs:
    submit_all += f"sbatch {job['sh']}\n"
submit_all += f"echo 'Hotovo! Sleduj: squeue -u $USER'\n"

submit_path = f"{EXP_DIR}/submit_all.sh"
with open(submit_path, "w") as f:
    f.write(submit_all)
os.chmod(submit_path, 0o755)

# Results collector
collector = f"""#!/usr/bin/env python3
import glob, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = []
for f in glob.glob("{EXP_DIR}/*/metrics.json"):
    with open(f) as fh:
        results.append(json.load(fh))

if not results:
    print("Žiadne výsledky ešte.")
    exit()

df = pd.DataFrame(results).sort_values(["fraction","lora_rank"])
print("\\n=== EXP 1: LEARNING CURVES ===")
print(df[["run_name","fraction","n_samples","lora_rank","eval_loss","perplexity"]].to_string(index=False))

# Graf
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for rank in df["lora_rank"].unique():
    sub = df[df["lora_rank"]==rank].sort_values("fraction")
    axes[0].plot(sub["fraction"]*100, sub["eval_loss"], marker="o", label=f"rank={{rank}}")
    axes[1].plot(sub["fraction"]*100, sub["perplexity"], marker="o", label=f"rank={{rank}}")

axes[0].set_xlabel("Dataset size (%)")
axes[0].set_ylabel("Eval Loss")
axes[0].set_title("Learning Curves – Eval Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Dataset size (%)")
axes[1].set_ylabel("Perplexity")
axes[1].set_title("Learning Curves – Perplexity")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("{EXP_DIR}/learning_curves.png", dpi=150)
print(f"\\nGraf uložený: {EXP_DIR}/learning_curves.png")
df.to_csv("{EXP_DIR}/results.csv", index=False)
"""

with open(f"{EXP_DIR}/collect_results.py", "w") as f:
    f.write(collector)

# Ulož summary
summary = {
    "experiment": "Learning Curves",
    "exp_id": EXP_ID,
    "exp_dir": EXP_DIR,
    "total_jobs": len(jobs),
    "fractions": DATASET_FRACTIONS,
    "lora_ranks": LORA_RANKS,
    "estimated_gpu_hours": len(jobs) * 4 * 8,
    "submit_script": submit_path,
    "results_collector": f"{EXP_DIR}/collect_results.py",
}
with open(f"{EXP_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n╔══════════════════════════════════════╗")
print(f"║      EXP 1 PRIPRAVENÝ!              ║")
print(f"╚══════════════════════════════════════╝")
print(f"Jobov:           {len(jobs)}")
print(f"Est. GPU hodín:  ~{len(jobs)*4*8:,}")
print(f"Exp adresár:     {EXP_DIR}")
print(f"\nSpusti joby:")
print(f"  bash {submit_path}")
print(f"\nPo dokončení:")
print(f"  python3 {EXP_DIR}/collect_results.py")
