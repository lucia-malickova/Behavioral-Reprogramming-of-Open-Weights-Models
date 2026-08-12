"""
Experiment 5: Epoch Ablation
Otázka: Kde je sweet spot pred overfittingom?
Checkpointy: po 1,2,3,5,7,10 epochách
Joby: 6 checkpoint × 3 LoRA ranky = 18 jobov
GPU hodiny: ~2,016h
"""

import os
import json
from datetime import datetime

BASE_OUTPUT = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/exp5_epoch_ablation"
ENV_PATH = "/leonardo_work/AIFAC_F02_159/env_finetuning"
DATASET_PATH = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/dataset/lucy_CLEAN_FINAL.jsonl"
BASE_MODEL = "/leonardo_scratch/large/userexternal/tbonomo0/models/meta-llama--Llama-3.1-8B-Instruct"
ACCOUNT = "AIFAC_F02_159"
PARTITION = "boost_usr_prod"
TIME = "12:00:00"

EPOCH_TARGETS = [1, 2, 3, 5, 7, 10]
LORA_RANKS = [8, 16, 32]
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4

EXP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"{BASE_OUTPUT}/exp_{EXP_ID}"

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)

LUCY_SYSTEM = "Si Lucy – digitálna dvojička Lucie Maličkovej. Si priama, empatická, vtipná keď je na to čas. Hovoríš prirodzene, bez kecania."

print(f"╔══════════════════════════════════════╗")
print(f"║   EXP 5: EPOCH ABLATION – SETUP     ║")
print(f"╚══════════════════════════════════════╝")

jobs = []
for epochs in EPOCH_TARGETS:
    for rank in LORA_RANKS:
        run_name = f"ea_e{epochs:02d}_r{rank:02d}"
        run_dir = f"{EXP_DIR}/{run_name}"
        os.makedirs(run_dir, exist_ok=True)

        py_script = f'''import os, json, math, torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

NUM_EPOCHS = {epochs}
LORA_RANK = {rank}
LORA_ALPHA = {rank * 2}
RUN_NAME = "{run_name}"
RUN_DIR = "{run_dir}"
MODEL_PATH = "{BASE_MODEL}"
DATASET_PATH = "{DATASET_PATH}"

print(f"START: {{RUN_NAME}} | epochs={{NUM_EPOCHS}} | rank={{LORA_RANK}}")

raw_data = []
with open(DATASET_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

LUCY_SYSTEM = "{LUCY_SYSTEM}"

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

# Loguj eval_loss po každej epoche
epoch_metrics = []

class EpochCallback:
    def on_epoch_end(self, args, state, control, **kwargs):
        pass

args = TrainingArguments(
    output_dir=RUN_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size={BATCH_SIZE},
    gradient_accumulation_steps={GRAD_ACCUM},
    learning_rate={LEARNING_RATE},
    weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_strategy="epoch",
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

# Zober eval_loss históriu
history = []
for log in trainer.state.log_history:
    if "eval_loss" in log:
        history.append({{
            "epoch": log.get("epoch"),
            "eval_loss": log["eval_loss"],
            "perplexity": math.exp(log["eval_loss"]),
        }})

metrics = {{
    "run_name": RUN_NAME,
    "num_epochs": NUM_EPOCHS,
    "lora_rank": LORA_RANK,
    "train_loss": result.metrics.get("train_loss"),
    "best_eval_loss": min(h["eval_loss"] for h in history) if history else None,
    "best_epoch": min(history, key=lambda x: x["eval_loss"])["epoch"] if history else None,
    "epoch_history": history,
    "runtime_min": result.metrics.get("train_runtime",0)/60,
}}

with open(f"{{RUN_DIR}}/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print(f"DONE {{RUN_NAME}}: best_eval_loss={{metrics['best_eval_loss']:.4f}} @ epoch={{metrics['best_epoch']}}")
'''

        py_path = f"{EXP_DIR}/scripts/{run_name}.py"
        with open(py_path, "w") as f:
            f.write(py_script)

        sbatch = f"""#!/bin/bash
#SBATCH --job-name=ea_{epochs}_{rank}
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
python3 {py_path}
"""
        sbatch_path = f"{EXP_DIR}/scripts/{run_name}.sh"
        with open(sbatch_path, "w") as f:
            f.write(sbatch)

        jobs.append({"name": run_name, "epochs": epochs, "rank": rank, "sh": sbatch_path})
        print(f"  ✅ {run_name}")

submit_all = f"#!/bin/bash\necho 'Submitujem {len(jobs)} epoch ablation jobov...'\n"
for job in jobs:
    submit_all += f"sbatch {job['sh']}\n"
submit_all += "echo 'Hotovo!'\n"

submit_path = f"{EXP_DIR}/submit_all.sh"
with open(submit_path, "w") as f:
    f.write(submit_all)
os.chmod(submit_path, 0o755)

collector = f"""#!/usr/bin/env python3
import glob, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = []
for f in glob.glob("{EXP_DIR}/*/metrics.json"):
    with open(f) as fh:
        data = json.load(fh)
        results.append({{k:v for k,v in data.items() if k != "epoch_history"}})

if not results:
    print("Žiadne výsledky ešte.")
    exit()

df = pd.DataFrame(results).sort_values(["lora_rank","num_epochs"])
print("\\n=== EXP 5: EPOCH ABLATION ===")
print(df[["run_name","num_epochs","lora_rank","best_eval_loss","best_epoch","runtime_min"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
for rank in sorted(df["lora_rank"].unique()):
    sub = df[df["lora_rank"]==rank].sort_values("num_epochs")
    ax.plot(sub["num_epochs"], sub["best_eval_loss"], marker="o", label=f"LoRA rank={{rank}}")

ax.set_xlabel("Max Epochs")
ax.set_ylabel("Best Eval Loss")
ax.set_title("Epoch Ablation – Finding the Sweet Spot")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("{EXP_DIR}/epoch_ablation.png", dpi=150)
print(f"Graf: {EXP_DIR}/epoch_ablation.png")
df.to_csv("{EXP_DIR}/results.csv", index=False)

# Odporuč najlepší počet epoch
best = df.nsmallest(1, "best_eval_loss").iloc[0]
print(f"\\n🏆 NAJLEPŠÍ: rank={{best['lora_rank']}}, epochs={{best['num_epochs']}}, eval_loss={{best['best_eval_loss']:.4f}}")
"""

with open(f"{EXP_DIR}/collect_results.py", "w") as f:
    f.write(collector)

print(f"\n✅ Jobov: {len(jobs)}")
print(f"Est. GPU hodín: ~{len(jobs)*4*12:,}")
print(f"Submit: bash {submit_path}")
